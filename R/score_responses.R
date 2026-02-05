#' Score open responses via multilingual spaCy similarity
#'
#' This helper propagates human scores to new responses by comparing
#' preprocessed texts with embeddings created through spaCy language models.
#' Any installed model can be supplied, making it straightforward to reuse the
#' same pipeline for Dutch, English, or other languages.
#'
#' @param response_data a data.frame or tibble with 3 columns (response_id, response, score) where response_id is the unique identifier for each response, response is the text of the response, and score is the score of the response given by a human rater (the response can be NA)
#' @param possible_item_scores a vector with the possible scores for the responses
#' @param preprocessing logical; if TRUE preprocess the text before calculating the similarity
#' @param spellcheck logical; if TRUE apply spellcheck the text before calculating the similarity
#' @param pos_keep a character vector with the parts of speech to keep
#' @param min_sim_weight a numeric value between 0 and 1; the minimum similarity weight to accept a similarity
#' @param n_highest_sim an integer value; the number of highest similarities above the min_sim_weight to accept
#' @param start_weight a numeric value between 0 and 1; the weight to start with for the similarity on the min_sim_weight
#' @param min_obs_accept an integer value; the minimum number of observations to accept a score
#' @param min_certainty_accept a numeric value between 0 and 1; the minimum certainty to accept a score
#' @param additonal_parameters a list of parameters to be passed to the function
#'
#' @returns The input response_data with additional feature, probability, and
#' expected-score columns.
#' @export
#' @examples
#' \dontrun{
#' library(tibble)
#'
#' responses <- tibble(
#'   response_id = 1:4,
#'   response = c(
#'     "De rivier trad buiten haar oevers.",
#'     "Het water stroomde door de straten.",
#'     "Water levels rise quickly.",
#'     "The levee holds."),
#'   score = c(2, 2, 1, 3)
#' )
#'
#' # Default Dutch model
#' score_responses(
#'   response_data = responses,
#'   possible_item_scores = 0:3
#' )
#'
#' # Switch to an English spaCy pipeline
#' score_responses(
#'   response_data = responses,
#'   possible_item_scores = 0:3,
#'   spacy_model = "en_core_web_md"
#' )
#' }
score_responses <- function(response_data,
                            possible_item_scores = NULL,
                            preprocessing = TRUE,
                            spellcheck = TRUE,
                            pos_keep = c("NOUN", "PROPN", "VERB", "ADJ", "ADV"),
                            min_sim_weight = 0.8,
                            n_highest_sim = 5,
                            start_weight = 0.5,
                            min_obs_accept = 5,
                            min_certainty_accept = 0.75,
                            additonal_parameters = list(preprocess_params = list(),
                                                        spellcheck_params = list()),
                            spacy_model = "nl_core_news_lg") {

  if (!is.character(spacy_model) || length(spacy_model) != 1 || is.na(spacy_model)) {
    stop("spacy_model must be a non-missing character scalar.")
  }

  cat(paste0("\n -- checking spaCy model ", spacy_model, " -- \n"))

  if (inherits(try(spacyr::spacy_initialize(model = spacy_model), silent = TRUE), "try-error")) {
    cat("\n -- installing spacy -- \n")
    spacyr::spacy_install(lang_models = spacy_model)
    cat(paste0("\n -- installing spacy model ", spacy_model, " -- \n"))
    spacyr::spacy_download_langmodel(spacy_model)
    spacyr::spacy_initialize(model = spacy_model)
  }

  # Load python
  reticulate::use_python(reticulate::py_config()$python, required = TRUE)

  cat(paste0("\n -- starting preprocessing --\n\n"))

  # Preprocess the text
  if (preprocessing) {
    if (length(additonal_parameters$preprocess_params) > 0) {
      response_data$response_processed <- preprocess_text(response_data$response, additonal_parameters$preprocess_params)
    } else {
      response_data$response_processed <- preprocess_text(response_data$response)
    }
  } else {
    response_data$response_processed <- response_data$response
  }

  # remove rows with empty processed responses

  cat(paste0("\n --we removed ", nrow(response_data |>
                                        dplyr::filter(.data$response_processed == "")), " responses that where empty before/after preprocessing --\n\n"))

  response_data <-
    response_data |>
    dplyr::filter(.data$response_processed != "") |>
    dplyr::mutate(doc_id = dplyr::row_number())

  cat(paste0("\n -- starting nlp pipeline --\n\n"))


  spacy_object <-
    suppressWarnings(spacyr::spacy_parse(response_data$response_processed,
                        tag = TRUE,
                        additional_attributes = c("is_oov", "vector"))) |>
    dplyr::mutate(doc_id = as.integer(readr::parse_number(.data$doc_id))) |>
    tibble::as_tibble()

  cat(paste0("\n -- nlp pipeline finished --> starting spellcheck --\n\n"))

  spacy_object <-
    spacy_object |>
    dplyr::mutate(spelling_updated = FALSE)

  # Spellcheck the text
  if (spellcheck) {
    if (length(additonal_parameters$spellcheck_params) > 0) {
      closest_results <- spellcheck_text(spacy_object, additonal_parameters$spellcheck_params)
    } else {
      closest_results <- spellcheck_text(spacy_object)
    }

    if (!is.null(closest_results)) {
      spacy_object <-
        spacy_object |>
        dplyr::rows_update(closest_results |>
                             dplyr::mutate(spelling_updated = TRUE),
                           by = c("doc_id", "sentence_id", "token_id"))
    }

  }

  cat(paste0("\n -- spellcheck finished --> starting probable score calculation --\n\n"))

  response_data <-
    response_data |>
      dplyr::left_join(spacy_object |>
                       dplyr::filter(.data$pos %in% pos_keep) |>
                       dplyr::group_by(.data$doc_id) |>
                       dplyr::summarise(features_token = paste0(.data$token, collapse = " "),
                                        features_lemma = paste0(.data$lemma, collapse = " "),
                                        features_vector = list(Reduce("+", .data$vector)),
                                        spellcheck_fixed = any(.data$spelling_updated)),
                     by = dplyr::join_by("doc_id")) |>
    dplyr::select(-"doc_id") |>
    tidyr::drop_na("features_token")


  if (is.null(possible_item_scores)) {
    possible_item_scores <- sort(unique(response_data$score))
  }

  response_data <-
    prop_score_sim(data = response_data,
                   possible_item_scores = possible_item_scores,
                   min_sim_weight = min_sim_weight,
                   n_highest_sim = n_highest_sim,
                   start_weight = start_weight,
                   min_obs_accept = min_obs_accept,
                   min_certainty_accept = min_certainty_accept)


  # Calculate the similarity


  spacyr::spacy_finalize()

  return(response_data)

}

#' Preprocess a character vector
#'
#' @param x a character vector
#' @param preprocess_params a list of parameters to be passed to the function, the default parameters are; lowercase = TRUE, rm_special_characters = TRUE, rm_qti = TRUE, rm_digits = FALSE, rm_stopwords = TRUE
#'
#' @returns a preprocessed character vector
#' @export
preprocess_text <- function(x,
                            preprocess_params = list()) {

  default_params <- list(
    lowercase = TRUE,
    rm_special_characters = TRUE,
    rm_qti = TRUE,
    rm_digits = FALSE,
    rm_stopwords = TRUE
  )

  for (param in names(default_params)) {
    if (param %in% names(preprocess_params)) {
      if (is.logical(preprocess_params[[param]])) {
        default_params[[param]] <- preprocess_params[[param]]
      }
    }
  }

  # rm_qti removes all characters that are the result of responses being made within a QTI environment
  if (default_params$rm_qti) {
    x <- stringr::str_replace_all(x, "<[^>]*>", " ")
    x <- stringr::str_replace_all(x, "[\r\n]", " ")
    x <- stringr::str_replace_all(x, "&nbsp;", " ")
  }

  # rm_special_characters removes all special characters and punctuation
  if (default_params$rm_special_characters) {
    x <- stringr::str_replace_all(x, "[^[:alnum:]]", " ")
  }

  # rm_digits removes all digits
  if (default_params$rm_digits) {
    x <- stringr::str_replace_all(x, "[[:digit:]]", " ")
  }

  # lowercase converts all characters to lowercase
  if (default_params$lowercase) {
    x <- stringr::str_to_lower(x)
  }

  # rm_stopwords removes all stopwords
  # if (default_params$rm_stopwords) {
  #   stopwords <- load_stopwords()
  #   x <- stringr::str_replace_all(x, paste0("\\b", stopwords, "\\b"), " ")
  # }

  # remove additional whitespacing
  x <- stringr::str_squish(x)

  # Preprocess the text
  return(x)
}

#' loading stopwords from the text file in the package
#' @returns a character vector with stopwords
load_stopwords <- function() {
  # Locate and load stopwords from the text file in the package
  stopwords_path <- system.file("extdata", "stopwords.txt", package = "nlirp")

  if (file.exists(stopwords_path)) {
    stopwords <- readLines(con = stopwords_path)
  } else {
    stop("Stopwords file not found!")
  }

  return(stopwords)
}

#' spellcheck a character vector
#'
#' @param spacy_object a character vector
#' @param spellcheck_params a list of parameters to be passed to the function, the default parameters are; spellcheck_closeness = .99, pos_to_check = c("NOUN", "PROPN", "VERB", "ADJ", "ADV"), pos_to_use = c("NOUN", "PROPN", "VERB", "ADJ", "ADV")
#'
#' @returns a preprocessed character vector
#' @export
spellcheck_text <- function(spacy_object,
                            spellcheck_params = list()) {

  default_spellcheck_params <- list(
    spellcheck_closeness = .8,
    pos_to_check = c("NOUN", "PROPN", "VERB", "ADJ", "ADV"),
    pos_to_use = c("NOUN", "PROPN", "VERB", "ADJ", "ADV")
  )

  for (param in names(default_spellcheck_params)) {
    if (param %in% names(spellcheck_params)) {
      if (param == "spellcheck_closeness" & is.numeric(spellcheck_params[[param]])) {
        default_spellcheck_params[[param]] <- spellcheck_params[[param]]
      } else {
        if (is.character(spellcheck_params[[param]])) {
          default_spellcheck_params[[param]] <- spellcheck_params[[param]]
        }
      }
    }
  }


  # Filter out the out of vocabulary words
  spellcheckthese <-
    spacy_object |>
    dplyr::filter(.data$is_oov,
                  .data$pos %in% default_spellcheck_params$pos_to_check)

  if (nrow(spellcheckthese) > 0) {
    # Get the unique in vocabulary words
    usethesetoreplace <-
      spacy_object |>
      dplyr::filter(!.data$is_oov,
                    .data$pos %in% default_spellcheck_params$pos_to_use) |>
      dplyr::distinct(.data$token, .data$lemma, .data$pos, .data$entity, .data$tag, .data$vector)

    closest_results <-
      find_closest_token(spellcheckthese, usethesetoreplace)

    if (is.null(closest_results)) {
      cat(paste0("\n -- no suitable replacements were found for the mispelled words --\n\n"))
      return(NULL)
    } else {
      cat(paste0("\n -- spellcheck found suitable replacements for ", nrow(closest_results), " out of ", nrow(spellcheckthese), " mispelled words--\n\n"))
      return(closest_results)
    }


  } else {
    cat(paste0("\n -- no mispelled words were found in the data --\n\n"))

    return(NULL)
  }

}

#' find closest token for mispoelled token
#'
#' @param misspelled_df a spacyr object with the tokens without a vector (incorrectly spelled)
#' @param correct_df a spacyr object with the tokens with a vector (correctly spelled)
#' @param accept_threshold the minimum similarity score to accept a match
#' @param close_match_treshold the minimum similarity score to accept a proposed match
#'
#' @returns a preprocessed character vector
find_closest_token <- function(misspelled_df, correct_df, accept_threshold = 0.8, close_match_treshold = 0.8) {
  results <- list()

  for (i in seq_len(nrow(misspelled_df))) {
    misspelled_entry <- misspelled_df[i, ]
    similarities <- numeric(nrow(correct_df))


    if (inherits(try(fuzzywuzzyR::GetCloseMatches(misspelled_entry$token,
                                                  correct_df$token,
                                                  n = 1,
                                                  cutoff = close_match_treshold), silent = TRUE), "try-error")) {
      proposed_entry <- list()

    } else {
      proposed_entry <- fuzzywuzzyR::GetCloseMatches(misspelled_entry$token,
                                                     correct_df$token,
                                                     n = 1,
                                                     cutoff = close_match_treshold)
    }

    for (j in seq_len(nrow(correct_df))) {
      correct_entry <- correct_df[j, ]

      token_sim <- 1 - stringdist::stringdist(misspelled_entry$token, correct_entry$token, method = "lv") / max(nchar(misspelled_entry$token), nchar(correct_entry$token))
      lemma_sim <- 1 - stringdist::stringdist(misspelled_entry$lemma, correct_entry$lemma, method = "lv") / max(nchar(misspelled_entry$lemma), nchar(correct_entry$lemma))
      tag_sim <- 1 - stringdist::stringdist(misspelled_entry$tag, correct_entry$tag, method = "lv") / max(nchar(misspelled_entry$tag), nchar(correct_entry$tag))

      pos_sim <- as.numeric(misspelled_entry$pos == correct_entry$pos)
      entity_sim <- as.numeric(misspelled_entry$entity == correct_entry$entity)


      ## dit kunnen we nog beter maken door proposed entry sim te matchen met kandidaat en dan overlap * close_match_treshold (via while loop)
      ## also update with weighted mean for different aspects to get more results (e.g., weight token + lemma_sim = 1, but tag & pos_sim = .5)
      if (purrr::is_empty(proposed_entry)) {
        similarities[j] <- sum(c(0.5, token_sim, lemma_sim, pos_sim, tag_sim, entity_sim))
      } else {
        proposed_entry_sim <- 1 - stringdist::stringdist(misspelled_entry$token, proposed_entry, method = "lv") / max(nchar(misspelled_entry$token), nchar(proposed_entry))
        similarities[j] <- sum(c(proposed_entry_sim, token_sim, lemma_sim, pos_sim, tag_sim, entity_sim))
      }

    }

    max_similarity <- max(similarities)
    best_match_index <- which.max(similarities)

    if (max_similarity > accept_threshold) {
      closest_match <- correct_df[best_match_index[1], "token"]
      results[[i]] <- misspelled_entry |>
        dplyr::select("doc_id", "sentence_id", "token_id", "is_oov") |>
        dplyr::bind_cols(correct_df[best_match_index, ] |>
                           dplyr::select("token", "lemma", "pos", "tag", "entity", "vector"))
    }
  }

  if (length(results) == 0) {
    return(NULL)
  } else {
    results |>
      dplyr::bind_rows() |>
      dplyr::relocate("is_oov", .after = "entity")
  }


}


utils::globalVariables(".") # to make sure the function is available in the global environment

#' calculate probabilities for possible scores based on similarity
#'
#' @param data a data.frame with the response_data after preprocesssing, spellcheck and vectorization with nlp
#' @param possible_item_scores a vector with the possible scores for the responses
#' @param min_sim_weight the minimum similarity weight to accept a similarity
#' @param n_highest_sim the number of highest similarities above the min_sim_weight to accept
#' @param start_weight the weight to start with for the similarity on the min_sim_weight
#' @param min_obs_accept the minimum number of observations to accept a score
#' @param min_certainty_accept the minimum certainty to accept a score
#'
#' @returns a data.frame with the expected scores for each response
prop_score_sim <- function(data,
                           possible_item_scores,
                           min_sim_weight = 0.8,
                           n_highest_sim = 5,
                           start_weight = 0.5,
                           min_obs_accept = 5,
                           min_certainty_accept = 0.75) {

  coss <- function(x) {crossprod(x)/(sqrt(tcrossprod(colSums(x^2))))}

  distance_mat <-
    data$features_vector %>%
    do.call(cbind, .) %>%
    coss(.)


  # zet alle NA/Nan cosine similarities op 0
  distance_mat[is.na(distance_mat)] = 0

  # optionally set all diagonals to 0 [for evaluation purposes]
  diag(distance_mat) = 0

  # creer een nieuwe matrix waarin alleen de similarities boven de min_sim_weight grens niet NA zijn
  min_distance_mat <- (distance_mat > min_sim_weight) * distance_mat
  min_distance_mat <- apply(min_distance_mat, 1, function(x) x * (x >= min(sort(x, decreasing = TRUE)[1:n_highest_sim], na.rm = TRUE)))
  min_distance_mat[min_distance_mat == 0] = NA

  # schaal nu de similarities naar gewichten zodat:
  # een similarity van min_sim_weight krijgt start_weight als gewicht
  # een similarity van 1 krijg 1 als gewicht.
  # alles daartussen in wordt linear omhoog geschaald [dus bij min_sim_weight = .8 en start_weight = .5 krijgt een cosine similarity van .9 het gewicht .75]
  min_distance_mat <- start_weight + ((min_distance_mat - min_sim_weight) / (1 - min_sim_weight) * (1 - start_weight))

  # creeer een score matrix waarbij alleen de scores voor antwoorden met similarities boven de min.dist niet NA zijn

  sc <- t(data$score * t(sign(distance_mat)) * sign(min_distance_mat))


  # tel per response hoeveel responses een similarity hebben op of boven de 5 hoogste similarities
  data$n_similar <-  apply(min_distance_mat, 1, function(row) sum(!is.na(row)))

  # bereken de expected score door een gewogen gemiddelde te nemen van alle scores met een hoog genoege similarity --> weighted.mean = sum((x * w)[w != 0]) / sum(w)
  data$weighted_pos_score <-
    pbapply::pblapply(seq_len(nrow(min_distance_mat)), function(i) {
      data.frame(pos_score = possible_item_scores,
                 n_score = 0,
                 weight = 0,
                 min_weight = 0,
                 prop_score = 0) |>
        dplyr::rows_update(data.frame(pos_score = sc[i, ],
                                      weight = min_distance_mat[, i]) |>
                             tidyr::drop_na() |>
                             dplyr::group_by(.data$pos_score) |>
                             dplyr::summarise(n_score = dplyr::n(),
                                              min_weight = ifelse(length(.data$weight) > 0, min(.data$weight), 0),
                                              weight = sum(.data$weight)) |>
                             dplyr::ungroup() |>
                             dplyr::mutate(prop_score = (.data$weight / sum(.data$weight))),
                           by = "pos_score")
    })

  data <-
    data |>
    tidyr::unnest("weighted_pos_score")

  data <-
    data |>
    dplyr::group_by(.data$response_id) |>
    dplyr::slice_max(.data$prop_score, n = 1, with_ties = FALSE) |>
    dplyr::rowwise() |>
    dplyr::mutate(exp_score = ifelse((.data$prop_score >= min_certainty_accept) & (.data$n_score >= min_obs_accept), .data$pos_score, NA)) |>
    dplyr::ungroup()


  return(data)

}

