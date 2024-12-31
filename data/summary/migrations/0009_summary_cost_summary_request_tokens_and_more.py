# Generated by Django 4.1.13 on 2024-11-24 22:26

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('summary', '0008_alter_summary_text'),
    ]

    operations = [
        migrations.AddField(
            model_name='summary',
            name='cost',
            field=models.FloatField(default=None, null=True),
        ),
        migrations.AddField(
            model_name='summary',
            name='request_tokens',
            field=models.IntegerField(default=None, null=True),
        ),
        migrations.AddField(
            model_name='summary',
            name='response_tokens',
            field=models.IntegerField(default=None, null=True),
        ),
    ]