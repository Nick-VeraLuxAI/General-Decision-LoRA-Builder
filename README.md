The General-Decision-LoRA-Builder project you shared is designed as a full pipeline for generating, enriching, balancing, cleaning, preparing, and training a domain-specific LoRA model focused on realistic human decision-making scenarios. Here’s a high-level summary of its purpose and workflow, based on the files you uploaded:
What General-Decision-LoRA-Builder Does
1. Generates realistic decision-making scenario data

    Using decision_bank_generator.py, it uses multiple local llama.cpp LLM servers to generate JSON decision scenarios from scratch.

    Using general-decision-generator.py, it loads an existing decision bank and queries an LLM server to generate detailed reasoning for each possible option, expanding the dataset.

2. Enhances and enriches the dataset

    decision_reasoning_enhancer.py improves reasoning quality and adds categories, importance scores based on situation keywords.

    decision_enricher.py appends synthetic, topic-distributed enriched data entries to augment the dataset diversity.

3. Cleans and balances the dataset

    decision_cleaner.py deduplicates and removes invalid or redundant entries.

    decision_balancer.py expands the dataset to a large target size (e.g., 100k samples), ensuring balanced distribution over types, sentiments, and edge cases.

    final_clean.py fills in missing metadata fields to prepare a consistent final dataset.

4. Prepares the dataset for training

    prepare_general_decision_dataset.py converts the JSONL dataset into LoRA-friendly training pairs (prompt + natural response), saving in HuggingFace dataset format.

5. Trains a LoRA fine-tuning model

    train_general_decision_lora.py uses PyTorch distributed training and PEFT to fine-tune a LoRA model on the prepared dataset.

6. Tests and verifies model outputs

    Tester.py loads the base model and LoRA weights to generate sample outputs from a prompt formatted similarly to training data, verifying quality and parsing.

Summary of Typical Workflow Order

    Generate base decision bank:

        Run decision_bank_generator.py or general-decision-generator.py to produce initial JSONL datasets.

    Enhance reasoning & add metadata:

        Run decision_reasoning_enhancer.py.

    Add synthetic enriched data:

        Run decision_enricher.py to add topic-focused entries.

    Clean & deduplicate:

        Run decision_cleaner.py.

    Balance & expand to target size:

        Run decision_balancer.py.

    Final cleaning & metadata filling:

        Run final_clean.py.

    Format dataset for LoRA training:

        Run prepare_general_decision_dataset.py.

    Train LoRA model:

        Run train_general_decision_lora.py.

    Test trained model:

        Run Tester.py.

What the project achieves

    Builds a large-scale, diverse, richly annotated dataset of decision scenarios.

    Trains a LoRA fine-tuned language model capable of generating human-like decision reasoning given a situation and options.

    The resulting LoRA can be integrated with a base model (e.g., CodeLLaMA-13b) for practical decision support or related AI applications.
