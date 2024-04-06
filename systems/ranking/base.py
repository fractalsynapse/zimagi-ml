from sklearn.metrics.pairwise import cosine_similarity

from utility.data import Collection, ensure_list
from utility.topics import TopicModel

import re
import math
import statistics
import copy


class BaseRanker(object):

    def __init__(self,
        command,
        instance_facade
    ):
        self.command = command

        self.topics = TopicModel()
        self.topic_index = {}
        self.topic_mean = 0

        self.instance_facade = instance_facade
        self.instance_filters = {
            'processed_time__isnull': False,
            'response_deadline__gte': self.command.time.now
        }
        self.instance_name_field = None
        self.instance_id_field = None
        self.group_id_field = None
        self.instance_collection = None

        self.instance_topics_field = 'topics'


    def generate(self,
        focus_cutoff_score = None,
        focus_selectivity = None,
        focus_limit = None,
        search_cutoff_score = None,
        search_selectivity = None,
        search_limit = None,
        **options
    ):
        if not self.instance_name_field:
            raise NotImplementedError("Instance field 'instance_name_field' must be set in subclass")
        if not self.instance_id_field:
            raise NotImplementedError("Instance field 'instance_id_field' must be set in subclass")
        if not self.instance_collection:
            raise NotImplementedError("Instance field 'instance_collection' must be set in subclass")

        search = self._generate_search(**options)
        if search:
            if search.text:
                search = self._rank_sentences(
                    search, search.text,
                    focus_cutoff_score,
                    focus_selectivity,
                    focus_limit
                )

            self.topic_index = self.topics.get_index(*search.sentences)
            if self.topic_index:
                self.topic_mean = statistics.mean(self.topic_index.values())

            self.command.info('')
            self.command.info('Search Sentences')
            self.command.info('-' * self.command.display_width)
            for sentence in search.sentences:
                self.command.info(" * {}".format(sentence))

            self.command.info('')
            self.command.info('Topic Index')
            self.command.info('-' * self.command.display_width)
            for topic, count in self.topic_index.items():
                if count >= self.topic_mean:
                    self.command.data(topic, count)

        return self._get_ranked_instances(
            self._filter(**options),
            search_selectivity,
            search_limit
        )
        # return self._rank_instances(
        #     search,
        #     instance_ids = self._filter(**options),
        #     cutoff_score = search_cutoff_score,
        #     selectivity = search_selectivity,
        #     search_limit = search_limit,
        #     focus_limit = focus_limit,
        #     **options
        # )


    def _generate_search(self,
        focus_text = None,
        **options
    ):
        search = None
        search_text = None

        if focus_text:
            search_text = self.command.generate_text_embeddings(focus_text, validate = False)
            if search_text:
                search = copy.deepcopy(search_text)

        if self.command.debug:
            if search_text:
                self.command.info('')
                self.command.info('Focus Sentences')
                self.command.info('-' * self.command.display_width)
                for sentence in search_text.sentences:
                    self.command.info(" * {}".format(sentence))

        return Collection(
            text = search_text,
            sentences = search.sentences,
            embeddings = search.embeddings
        ) if search else None


    def _filter(self, **options):
        return list(self._filter_instances(**options).values_list('id', flat = True))

    def _filter_instances(self, **options):
        instance_query = self.instance_facade.filter(**self.instance_filters)
        if self.topic_index:
            instance_query = instance_query.filter(
                **{ "{}__has_any_keys".format(self.instance_topics_field): list(self.topic_index.keys()) }
            )
        return instance_query


    def _rank_instances(self,
        search,
        instance_ids,
        cutoff_score,
        selectivity,
        search_limit,
        focus_limit,
        **options
    ):
        instance_data = self._generate_ranking(
            search = search,
            instance_ids = instance_ids,
            cutoff_score = cutoff_score,
            focus_limit = focus_limit,
            **options
        )
        return self._filter_recommendations(
            self._calculate_scores(search, instance_data),
            selectivity,
            search_limit
        )


    def _generate_ranking(self,
        search,
        instance_ids,
        cutoff_score,
        focus_limit,
        **options
    ):
        instance_data = Collection(
            ids = instance_ids,
            cutoff_score = cutoff_score,
            focus_limit = focus_limit,
            scores = {},
            counts = {},
            index = {}
        )

        if search:
            instance_rankings = self._search_embeddings(
                self.instance_collection,
                search.embeddings,
                filter_field = self.instance_id_field,
                filter_ids = instance_data.ids,
                result_fields = self.group_id_field if self.instance_id_field != self.group_id_field else None,
                limit = ((instance_data.focus_limit / len(search.sentences)) * 10)
            )
            for index, ranking in enumerate(instance_rankings):
                if self.command.debug:
                    self.command.info('-' * self.command.display_width)
                    self.command.info(" * [ {} ] - {}".format(index, search.sentences[index]))
                    self.command.info('')

                self._score_instances(instance_data, index, ranking)

        return instance_data

    def _score_instances(self, instance_data, index, ranking):
        for ranking_index, sentence_info in enumerate(ranking):
            if self.instance_id_field in sentence_info.payload and self.group_id_field in sentence_info.payload:
                group_id = sentence_info.payload[self.group_id_field]
                instance_id = sentence_info.payload[self.instance_id_field]
                sentence = sentence_info.payload['sentence']

                if group_id not in instance_data.index:
                    instance_data.index[group_id] = {}

                if sentence not in instance_data.index[group_id]:
                    topic_score = self._get_topic_score(sentence)

                    if len(re.split(r'\s+', sentence.strip())) >= 5 \
                        and topic_score >= self.topic_mean \
                        and sentence_info.score >= instance_data.cutoff_score:

                        if instance_id not in instance_data.scores:
                            instance_data.scores[instance_id] = sentence_info.score
                            instance_data.counts[instance_id] = 1
                        else:
                            instance_data.scores[instance_id] += sentence_info.score
                            instance_data.counts[instance_id] += 1

                        if self.command.debug:
                            self.command.info(" * [ {} ] - {} ( {} )".format(instance_id, sentence_info.payload['sentence'], sentence_info.score))

                    instance_data.index[group_id][sentence] = True


    def _calculate_scores(self, search, instance_data):
        if search:
            search_total = (instance_data.focus_limit * 10)

            for instance_id, instance_score in instance_data.scores.items():
                instance_data.scores[instance_id] = (
                    math.sqrt(
                        (instance_score / instance_data.counts[instance_id])
                        * (instance_data.counts[instance_id] / search_total)
                    ) * 100
                )
                if self.command.debug:
                    self.command.info("SQRT(({} / {}) * ({} / {})) * 100 = {}  [ {} ]".format(
                        round(instance_score, 2),
                        instance_data.counts[instance_id],
                        instance_data.counts[instance_id],
                        search_total,
                        round(instance_data.scores[instance_id], 2),
                        instance_id
                    ))

        elif instance_data.ids:
            for instance_id in instance_data.ids:
                instance_data.scores[instance_id] = -1

        return instance_data.scores


    def _get_ranked_instances(self, instance_ids):
        scores = {}

        for instance_id in instance_ids:
            instance = self.instance_facade.retrieve_by_id(instance_id)
            scores[instance_id] = self._get_instance_score(instance)

        return self._filter_recommendations(scores, selectivity, search_limit)

    def _get_instance_score(self, instance):
        score = 0
        if instance.topics:
            for topic, count in instance.topics.items():
                if topic in self.topic_index:
                    score += (count * self.topic_index[topic])
        return score


    def _filter_recommendations(self, scores, selectivity, search_limit):
        ranked_instances = []
        score_values = scores.values()
        cutoff_score = statistics.mean(score_values) + (selectivity * statistics.stdev(score_values)) if selectivity is not None and len(score_values) > 1 else 0
        index = 1

        if self.command.debug:
            self.command.info('')
            self.command.info('Ranked Instances')
            self.command.data('Cutoff Score', cutoff_score)
            self.command.info('-' * self.command.display_width)

        for instance_id, score in sorted(scores.items(), key = lambda x: float(x[1]), reverse = True):
            instance = self.instance_facade.retrieve_by_id(instance_id)
            if instance and score >= cutoff_score:
                ranked_instances.append(Collection(
                    score = score,
                    instance = instance
                ))
                if search_limit and index == search_limit:
                    break
                index += 1

                if self.command.debug or self.command.verbosity == 3:
                    self.command.info(" [ {} ] {} ({})".format(
                        self.command.success_color(round(score, 2)),
                        self.command.key_color(getattr(instance, self.instance_name_field)),
                        self.command.value_color(instance.id)
                    ))
                elif self.command.verbosity > 1:
                    self.command.info(" [ {} ] {}".format(
                        self.command.success_color(round(score, 2)),
                        self.command.key_color(getattr(instance, self.instance_name_field))
                    ))
        return ranked_instances


    def _rank_sentences(self, sentence_data, context_data, cutoff_score, selectivity, limit):
        rankings = {}
        sentences = []
        embeddings = []
        scores = []
        topic_index = self.topics.filtered_index(sentence_data.sentences, context_data.sentences)
        similarities = cosine_similarity(context_data.embeddings, sentence_data.embeddings)

        # Focus text
        for context_index, context_rankings in enumerate(similarities):
            # Full sentences
            for sentence_index, score in enumerate(context_rankings):
                if score >= cutoff_score:
                    if sentence_index not in rankings:
                        rankings[sentence_index] = [ float(score) ]
                    else:
                        rankings[sentence_index].append(float(score))

        for sentence_index, sentence_scores in rankings.items():
            topic_score = self._get_topic_score(sentence_data.sentences[sentence_index], topic_index)
            rankings[sentence_index] = (topic_score * max([ *sentence_scores, 0 ]) * statistics.mean(sentence_scores))

        values = rankings.values()
        stdev = statistics.stdev(values) if len(values) > 1 else 0
        selectivity_cutoff_score = statistics.mean(values) + (selectivity * stdev) if selectivity is not None else 0

        for sentence_index, score in sorted(rankings.items(), key = lambda x: float(x[1]), reverse = True):
            if score >= selectivity_cutoff_score:
                sentences.append(sentence_data.sentences[sentence_index])
                embeddings.append(sentence_data.embeddings[sentence_index])
                scores.append(score)

                if len(sentences) == limit:
                    break

        return Collection(
            sentences = sentences,
            embeddings = embeddings,
            scores = scores
        )


    def _get_topic_score(self, sentence, topic_index = None):
        if not topic_index:
            topic_index = self.topic_index

        topic_score = 0
        for topic in self.topics.parse(sentence):
            if topic in topic_index:
                topic_score += topic_index[topic]

        return topic_score


    def _get_embeddings(self, collection, id_field, ids):
        if not ids:
            return None
        return self.command.get_embeddings(
            collection,
            **{ id_field: ids }
        )

    def _search_embeddings(self, collection, embeddings, filter_field, filter_ids = None, result_fields = None, limit = 10):
        qdrant = self.command.qdrant(collection)
        options = {
            'limit': limit,
            'fields': [
                *ensure_list(result_fields if result_fields else []),
                filter_field,
                'sentence'
            ]
        }
        if filter_ids is not None:
            if not len(filter_ids):
                return []

            options['filter_field'] = filter_field
            options['filter_values'] = filter_ids

        return qdrant.search(embeddings, **options)


    def _request_webpage(self, url):
        web_info = Collection(
            url = url,
            text = None
        )
        if url and '.' in url:
            base_url = re.sub(r'^https?\:\/\/', '', url, flags = re.IGNORECASE)

            for protocol in [ 'https', 'http' ]:
                url = "{}://{}".format(protocol, base_url)

                self.command.data('Requesting vendor URL', url)
                webpage = self.command.parse_webpage(url)
                if webpage.text:
                    web_info.url = webpage.url
                    web_info.text = webpage.text
                    break

        return web_info
