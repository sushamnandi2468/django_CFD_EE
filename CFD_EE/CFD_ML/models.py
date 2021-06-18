from django.db import models
from django.utils import timezone

# Create your models here.

class FraudDetectCount(models.Model):
    Filename = models.CharField(max_length=100)
    Percentage = models.FloatField(max_length=50)
    FalsePos = models.IntegerField()
    Suspicious = models.IntegerField()
    NewCust = models.IntegerField()
    DateTime = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return self.Filename 