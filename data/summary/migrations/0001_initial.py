# Generated by Django 4.1.13 on 2024-02-24 18:00

from django.db import migrations, models
import systems.models.fields


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Summary',
            fields=[
                ('created', models.DateTimeField(editable=False, null=True)),
                ('updated', models.DateTimeField(editable=False, null=True)),
                ('id', models.CharField(editable=False, max_length=64, primary_key=True, serialize=False)),
                ('text', models.TextField(default=None, null=True)),
                ('result', models.TextField(default=None, null=True)),
                ('config', systems.models.fields.DictionaryField(default=dict)),
                ('stats', systems.models.fields.DictionaryField(default=dict)),
            ],
            options={
                'verbose_name': 'summary',
                'verbose_name_plural': 'summaries',
                'db_table': 'ml_summary',
                'ordering': ['id'],
                'abstract': False,
                'unique_together': {('text', 'config')},
            },
        ),
    ]