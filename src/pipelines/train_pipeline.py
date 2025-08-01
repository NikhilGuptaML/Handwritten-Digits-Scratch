from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainPipeline:
    def __init__(self):
        self.ingestor = DataIngestion()
        self.transformer = DataTransformation()
        self.trainer = ModelTrainer()

    def run_pipeline(self):
        self.ingestor.initiate_unzipping_data()
        self.ingestor.initiate_unpacking_data()

        data_dict = self.transformer.get_data_transform_object()
        self.transformer.initiate_data_transformation(data_dict)

        final_data_dict = self.trainer.load_training_data()
        self.trainer.initiate_model_training(final_data_dict)
        

