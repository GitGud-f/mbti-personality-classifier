from src.preprocessing.morphology import MorphologyPreprocessor
from src.models.baseline_model import BinaryBaselineRunner
from src.features.syntax import SyntaxFeatureExtractor
from src.models.hybrid_model import HybridModelRunner
from src.preprocessing.semantics import OntologyPreprocessor
from src.features.topic_modeling import TopicModeler
from src.models.topic_model_runner import TopicModelRunner
from src.features.embeddings import EmbeddingExtractor
from src.models.embedding_model import EmbeddingModelRunner
from src.features.word2vec_self import Word2VecTrainer
from src.models.word2vec_runner import Word2VecRunner
from src.models.bert_finetune import BertFineTuner
import os

def run_stage_morphology():
    # 1. Initialize Processor
    morph = MorphologyPreprocessor()
    
    # --- Experiment A: Stemming ---
    if not os.path.exists("data/interim/mbti_stemmed.csv"):
        print("Generating Stemmed Data...")
        morph.run(method='stemming')
    
    runner = BinaryBaselineRunner()
    runner.train(
        data_path="data/interim/mbti_stemmed.csv", 
        experiment_name="02_Stemming_LogReg"
    )

    # --- Experiment B: Lemmatization ---
    if not os.path.exists("data/interim/mbti_lemmatized.csv"):
        print("Generating Lemmatized Data...")
        morph.run(method='lemmatization')

    runner.train(
        data_path="data/interim/mbti_lemmatized.csv", 
        experiment_name="02_Lemmatization_LogReg"
    )
    
def run_stage_syntax():
    # 1. Extract Features (Heavy computation)
    if not os.path.exists("data/processed/mbti_with_syntax.csv"):
        syntax_extractor = SyntaxFeatureExtractor()
        # Input should be the best text version (usually Lemmatized)
        syntax_extractor.run(input_path="data/interim/mbti_lemmatized.csv")
    
    # 2. Train Hybrid Model
    hybrid_runner = HybridModelRunner()
    hybrid_runner.train(
        data_path="data/processed/mbti_with_syntax.csv", 
        experiment_name="03_Syntax_Hybrid"
    )

def run_stage_semantics():
    # 1. Generate Hypernym Data
    if not os.path.exists("data/interim/mbti_hypernyms.csv"):
        sem_processor = OntologyPreprocessor()
        # Input: Lemmatized data is best for WordNet lookups
        sem_processor.run(input_path="data/interim/mbti_lemmatized.csv")
    
    # 2. Train Model on Hypernyms
    runner = BinaryBaselineRunner()
    runner.train(
        data_path="data/interim/mbti_hypernyms.csv", 
        experiment_name="04_Semantics_Hypernyms"
    )

def run_stage_topics():
    # 1. Train LDA and Generate Features
    if not os.path.exists("data/processed/mbti_with_topics.csv"):
        # We use lemmatized data because LDA hates noise
        topic_modeler = TopicModeler(num_topics=20) 
        topic_modeler.run(input_path="data/interim/mbti_lemmatized.csv")
    
    # 2. Train Hybrid Model (TFIDF + Topics)
    runner = TopicModelRunner()
    runner.train(
        data_path="data/processed/mbti_with_topics.csv", 
        experiment_name="05_Topic_Modeling_LDA"
    )

def run_stage_embeddings():
    # 1. Generate GloVe Vectors
    if not os.path.exists("data/processed/mbti_glove_100.csv"):
        embedder = EmbeddingExtractor()
        # We can run this on the basic cleaned text
        embedder.run(input_path="data/interim/mbti_cleaned_basic.csv")
    
    # 2. Train Model
    runner = EmbeddingModelRunner()
    runner.train(
        data_path="data/processed/mbti_glove_100.csv", 
        experiment_name="06_GloVe_Embeddings"
    )

def run_stage_word2vec():
    # 1. Train Word2Vec and Extract
    if not os.path.exists("data/processed/mbti_word2vec_own.csv"):
        trainer = Word2VecTrainer()
        trainer.train_and_extract(input_path="data/interim/mbti_cleaned_basic.csv")
    
    # 2. Train Model
    runner = Word2VecRunner()
    runner.train(
        data_path="data/processed/mbti_word2vec_own.csv", 
        experiment_name="07_Word2Vec_SelfTrained"
    )

def run_stage_bert():
    # Warning for the user
    print("!!! WARNING: BERT Fine-Tuning is computationally intensive !!!")
    print("This stage might take 30-60 minutes per target on a standard GPU.")
    
    tuner = BertFineTuner()
    tuner.run(input_path="data/interim/mbti_cleaned_basic.csv")
    
if __name__ == "__main__":
    run_stage_morphology()
    run_stage_syntax()
    run_stage_semantics()
    run_stage_topics()
    run_stage_embeddings()
    run_stage_word2vec()
    run_stage_bert()