import os
import yaml
import pandas as pd
import logging
import dspy
from dspy.evaluate.metrics import answer_exact_match
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from scripts.utils import process_df, get_history
from scripts.predictor import Classification

GREEN = "\033[92m"
RESET = "\033[0m"
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load configuration files
def load_config(file_path):
    try:
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logging.error(f"Error loading configuration from {file_path}: {e}")
        raise

# Configure the language model
def configure_lm(model_config):
    try:
        lm = dspy.LM(model_config['model_name'],
                 api_base=model_config['api_base'],
                 api_key=model_config['api_key'],
                 model_type='text')
        dspy.configure(lm=lm)
        return lm
    except Exception as e:
        logging.error(f"Error configuring language model: {e}")
        raise

# Load and process the dataset
def load_dataset(data_config):
    try:
        train_df = pd.read_csv(os.path.join(data_config["data_dir"], 'train.csv'))
        test_df = pd.read_csv(os.path.join(data_config["data_dir"], 'test.csv'))

        processed_train_df = process_df(train_df, data_config['input_columns'], data_config['output_column'])
        processed_test_df = process_df(test_df, data_config['input_columns'], data_config['output_column'])

        train_set = [dspy.Example(question=row['question'], choices = row['choices'], answer=row['answer']).with_inputs("question","choices") for _, row in processed_train_df.iterrows()]
        test_set = [dspy.Example(question=row['question'], choices = row['choices'], answer=row['answer']).with_inputs("question","choices") for _, row in processed_test_df.iterrows()]

        logging.info(f'Loaded {len(train_set)} training examples and {len(test_set)} testing examples.')
        return train_set, test_set
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise

def main():
    # Load configurations
    data_config = load_config('config/data_config.yaml')
    model_config = load_config('config/model_config.yaml')

    # Configure language model
    lm = configure_lm(model_config)

    # Load and process dataset
    logging.info(f"{GREEN}Loading dataset...{RESET}")
    train_set, test_set = load_dataset(data_config)

    # Define classifier
    classifier = Classification()

    # Test initial prediction
    initial_output = classifier(question=test_set[0]['question'], choices=test_set[0]['choices'])
    logging.info(f'{GREEN}Initial test prediction (before optimization):{RESET} {initial_output}')

    # Evaluate the classifier
    logging.info(f'{GREEN}Evaluating program...{RESET}')
    evaluator = dspy.Evaluate(devset=test_set, metric=answer_exact_match, 
                              num_threads=8, display_progress=True, 
                              provide_traceback=True)
    eval_result = evaluator(classifier)
    logging.info(f'{GREEN}Evaluation result (zero-shot classification):{RESET} {eval_result}')

    # Optimize the classifier
    logging.info(f'{GREEN}Optimizing program...{RESET}')
    teleprompter_fsrs = BootstrapFewShotWithRandomSearch(metric=answer_exact_match, 
                                                         max_labeled_demos=16,
                                                         num_threads=8,
                                                         max_bootstrapped_demos=2,
                                                         num_candidate_programs=8,
                                                         max_rounds = 5
                                                         )
    optimized_classifier = teleprompter_fsrs.compile(classifier, trainset=train_set)

        # Evaluate the optimized classifier
    new_eval_result = evaluator(optimized_classifier)
    logging.info(f'{GREEN}Evaluation result (few-shot classification):{RESET} {new_eval_result}')

    # New test prediction
    new_output = optimized_classifier(question=test_set[0]['question'], choices=test_set[0]['choices'])
    logging.info(f'{GREEN}Final test prediction (after optimization):{RESET} {new_output}')
    
    # Save the optimized program
    logging.info(f'{GREEN}Saving optimized program...{RESET}')
    save_path = f'{data_config["save_path"]}/program_checkpoint.json'
    optimized_classifier.save(save_path)

    # Print prompt history
    logging.info(f'{GREEN}Latest prompt history:{RESET}')
    history = get_history(lm, 1)
    logging.info(history)

if __name__ == '__main__':
    main()
