import dpnp
import dpctl
from src.pipelines.train_pipeline import TrainPipeline
from src.pipelines.test_pipeline import TestPipeline


def train():
    pipeline = TrainPipeline()
    pipeline.run_pipeline()

def check_accuracy():
    test=TestPipeline()
    test.run_pipeline()

if __name__=="__main__":
    train()
   