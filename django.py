from django.conf import settings

from settings.config import Config

#
# Model cache paths
#
settings.PROJECT_PATH_MAP['st_model_cache'] = {
    'directory': 'st_models',
    'backup': False
}
settings.PROJECT_PATH_MAP['tr_model_cache'] = {
    'directory': 'tr_models',
    'backup': False
}
settings.PROJECT_PATH_MAP['hf_cache'] = {
    'directory': 'hf_cache',
    'backup': False
}

#
# OCR Processing
#
PDF_OCR_BATCH_SIZE = Config.integer('ZIMAGI_PDF_OCR_BATCH_SIZE', 10)
PDF_OCR_DPI = Config.integer('ZIMAGI_PDF_OCR_DPI', 200)

#
# ML Configurations
#
SENTENCE_PARSER_PROVIDERS = Config.list('ZIMAGI_SENTENCE_PARSER_PROVIDERS', [ 'core_en_web' ])
ENCODER_PROVIDERS = Config.list('ZIMAGI_ENCODER_PROVIDERS', [ 'mpnet_di' ])
SUMMARIZER_PROVIDERS = Config.list('ZIMAGI_SUMMARIZER_PROVIDERS', [ 'mixtral_di_7bx8' ])

SUMMARIZER_COST_PER_TOKEN = Config.decimal('ZIMAGI_SUMMARIZER_COST_PER_TOKEN', 0.0000003)

#
# HuggingFace Account
#
HUGGINGFACE_TOKEN = Config.string('ZIMAGI_HUGGINGFACE_TOKEN')

#
# DeepInfra Account
#
DEEPINFRA_API_KEY = Config.string('ZIMAGI_DEEPINFRA_API_KEY')

#
# Google Cloud Configurations
#
GOOGLE_SERVICE_CREDENTIALS = Config.string('ZIMAGI_GOOGLE_SERVICE_CREDENTIALS')
GOOGLE_VERTEX_AI_REGION = Config.string('ZIMAGI_GOOGLE_VERTEX_AI_REGION', 'us-central1')
GOOGLE_VERTEX_AI_BUCKET = Config.string('ZIMAGI_GOOGLE_VERTEX_AI_BUCKET')
