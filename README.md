# nlirp

`nlirp` scores open-ended responses in multiple languages by matching each response against previously rated answers and calculating an expected score based on semantic similarity. It wraps a lightweight workflow around `spacyr`, `reticulate`, and spaCy language models—defaulting to the Dutch `nl_core_news_lg` pipeline while allowing any installed model such as `en_core_web_md`—to handle text cleaning, optional spellchecking, vectorisation, and weighted similarity scoring.

## Features
- Semantic similarity scoring that mirrors human ratings using cosine similarity over spaCy embeddings in any spaCy-supported language for packages that contain word vectors.
- Optional preprocessing and spellchecking steps that can be customised through parameter lists.
- Weighted aggregation of the highest-confidence neighbour scores with tunable acceptance thresholds.
- Verbose console logging so you can follow each stage of the pipeline when running batch jobs.

## Installation
```r
# install.packages("remotes")
remotes::install_github("Citolab/nlirp")
```

The package depends on Python with spaCy and whichever language model you intend to use (defaults to `nl_core_news_lg`). The first call to `score_responses()` attempts to initialise spaCy and will prompt to install the core library and model if they are missing. For reproducible environments you can set them up manually:

```bash
python -m venv venv
source venv/bin/activate
pip install spacy
python -m spacy download nl_core_news_lg  # replace with e.g. en_core_web_md
```

Ensure that `reticulate::use_python()` can discover the Python executable that owns the model. You can set `RETICULATE_PYTHON` in your `.Renviron` if needed.

To use a different language model, download it with `python -m spacy download <model-name>` and pass the same name to `score_responses(spacy_model = "model-name")`. You can mix languages within a single dataset as long as the chosen model provides vectors for the responses you want to score.

## Quick start
```r
library(dplyr)
library(nlirp)

responses <- tibble(
	response_id = 1:6,
	response = c(
		"De rivier trad buiten haar oevers.",
		"Het water stroomde door de straten.",
		"Ik heb geen idee.",
		NA,
		"Het waterpeil was hoger dan normaal.",
		"We bouwden dijken om het water tegen te houden."
	),
	score = c(2, 2, 0, 1, 2, 3)
)

scored <- score_responses(
	response_data = responses,
	possible_item_scores = 0:3,
	preprocessing = TRUE,
	spellcheck = TRUE,
	min_sim_weight = 0.8,
	n_highest_sim = 5,
	start_weight = 0.5,
	min_obs_accept = 3,
	min_certainty_accept = 0.7
)

scored %>%
	select(response_id, response, exp_score, prop_score, n_similar)
```

The output augments the original data with:
- `response_processed`: the cleaned text after optional preprocessing.
- `features_token`, `features_lemma`, `features_vector`: token summaries used for similarity calculations.
- `weighted_pos_score`: per-response probabilities for each possible score.
- `exp_score`: the accepted expected score (or `NA` if certainty or neighbour counts are below thresholds).

## Customising preprocessing and spellchecking
You can pass nested parameter lists via `additonal_parameters` to fine-tune the workflow:

```r
score_responses(
	response_data = responses,
	additonal_parameters = list(
		preprocess_params = list(
			lowercase = TRUE,
			rm_special_characters = TRUE,
			rm_stopwords = FALSE
		),
		spellcheck_params = list(
			spellcheck_closeness = 0.85,
			pos_to_check = c("NOUN", "VERB")
		)
	)
)
```

Helper functions are exported for standalone use:
- `preprocess_text()` for configurable string normalisation.
- `spellcheck_text()` to repair out-of-vocabulary tokens detected by spaCy.

## Switching spaCy models
```r
score_responses(
	response_data = responses,
	spacy_model = "en_core_web_md"
)
```
Use this parameter to align the embeddings with the language of your responses. Any spaCy model available in your Python environment can be supplied, making it straightforward to support Dutch, English, or other locales within the same scoring workflow.

### Dutch vs. English example
```r
mixed_responses <- tibble(
	response_id = 1:4,
	response = c(
		"De dijk brak tijdens de storm.",
		"Water stroomde de polder in.",
		"De rivier overstroomde het land.",
		"Ingenieurs versterkten de dijk."),
	score = c(3, 2, 2, 3)
)

# Score Dutch answers with the default model
score_responses(mixed_responses, possible_item_scores = 0:3)

# Translate responses and evaluate with an English spaCy pipeline
english_responses <- tibble(
	response_id = mixed_responses$response_id,
	response = c(
		"The dike broke during the storm.",
		"Water flowed into the polder.",
		"The river flooded the fields.",
		"Engineers reinforced the levee."),
	score = mixed_responses$score
)

score_responses(
	response_data = english_responses,
	possible_item_scores = 0:3,
	spacy_model = "en_core_web_md"
)
```

## Tips
- Adjust `min_sim_weight`, `n_highest_sim`, and `start_weight` together to balance inclusiveness versus noise.
- Monitor `prop_score` and `n_similar` to detect responses with weak neighbours and consider manual review.

## Contributing
Issues and pull requests are welcome at [Citolab/nlirp](https://github.com/Citolab/nlirp). Please include reproducible examples and session info so we can troubleshoot quickly.

## License
Distributed under the GNU Lesser General Public License v2.1. See [LICENSE](LICENSE) for details.
