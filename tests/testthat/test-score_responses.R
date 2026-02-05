test_that("score_responses validates spaCy model input", {
  response_data <- tibble::tibble(
    response_id = 1,
    response = "Voorbeeld antwoord",
    score = 1
  )

  expect_error(
    score_responses(response_data, spacy_model = NA_character_),
    "spacy_model must be"
  )
})

test_that("preprocess_text respects configuration flags", {
  input <- c("<p>Rivier</p> 123")
  output <- preprocess_text(
    input,
    preprocess_params = list(
      lowercase = FALSE,
      rm_special_characters = FALSE,
      rm_qti = TRUE,
      rm_digits = TRUE
    )
  )

  expect_equal(output, "Rivier")
})

test_that("load_stopwords returns package stopwords", {
  stops <- load_stopwords()
  expect_type(stops, "character")
  expect_gt(length(stops), 0)
})

test_that("score_responses aggregates multiple responses", {
  responses <- tibble::tibble(
    response_id = 1:2,
    response = c("Waterveiligheid", "Waterveiligheid"),
    score = c(1, 1)
  )

  fake_parse <- function(text, ...) {
    tibble::tibble(
      doc_id = c("doc1", "doc2"),
      sentence_id = c(1L, 1L),
      token_id = c(1L, 1L),
      token = c("veiligheid", "veiligheid"),
      lemma = c("veiligheid", "veiligheid"),
      pos = c("NOUN", "NOUN"),
      entity = c("", ""),
      tag = c("NOUN", "NOUN"),
      vector = list(rep(1, 3), rep(1, 3)),
      is_oov = c(FALSE, FALSE)
    )
  }

  fake_init <- function(model) invisible(TRUE)
  fake_install <- function(lang_models) invisible(TRUE)
  fake_download <- function(model) invisible(TRUE)
  fake_finalize <- function() invisible(TRUE)
  fake_use_python <- function(...) invisible(TRUE)
  fake_py_config <- function() list(python = "/usr/bin/python")

  testthat::local_mocked_bindings(
    use_python = fake_use_python,
    py_config = fake_py_config,
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(
    spacy_initialize = fake_init,
    spacy_install = fake_install,
    spacy_download_langmodel = fake_download,
    spacy_parse = fake_parse,
    spacy_finalize = fake_finalize,
    .package = "spacyr"
  )

  result <- score_responses(
    response_data = responses,
    possible_item_scores = 0:1,
    preprocessing = FALSE,
    spellcheck = FALSE,
    min_obs_accept = 1,
    min_certainty_accept = 0.5,
    spacy_model = "nl_core_news_lg"
  )

  expect_equal(nrow(result), 2)
  expect_true(all(result$exp_score == 1))
})

test_that("score_responses initialises requested spaCy models", {
  responses <- tibble::tibble(
    response_id = 1:2,
    response = c("Het water stijgt", "The levee holds"),
    score = c(1, 1)
  )

  fake_output <- tibble::tibble(
    doc_id = c("doc1", "doc2"),
    sentence_id = c(1L, 1L),
    token_id = c(1L, 1L),
    token = c("water", "levee"),
    lemma = c("water", "levee"),
    pos = c("NOUN", "NOUN"),
    entity = c("", ""),
    tag = c("NOUN", "NOUN"),
    vector = list(rep(1, 3), rep(1, 3)),
    is_oov = c(FALSE, FALSE)
  )

  parse_count <- 0L
  captured_models <- character()

  fake_parse <- function(text, ...) {
    parse_count <<- parse_count + 1L
    fake_output
  }

  fake_init <- function(model) {
    captured_models <<- c(captured_models, model)
    invisible(TRUE)
  }

  fake_install <- function(lang_models) invisible(TRUE)
  fake_download <- function(model) invisible(TRUE)
  fake_finalize <- function() invisible(TRUE)
  fake_use_python <- function(...) invisible(TRUE)
  fake_py_config <- function() list(python = "/usr/bin/python")

  testthat::local_mocked_bindings(
    use_python = fake_use_python,
    py_config = fake_py_config,
    .package = "reticulate"
  )

  testthat::local_mocked_bindings(
    spacy_initialize = fake_init,
    spacy_install = fake_install,
    spacy_download_langmodel = fake_download,
    spacy_parse = fake_parse,
    spacy_finalize = fake_finalize,
    .package = "spacyr"
  )

  score_responses(
    response_data = responses,
    possible_item_scores = 0:1,
    preprocessing = FALSE,
    spellcheck = FALSE,
    min_obs_accept = 1,
    min_certainty_accept = 0.5,
    spacy_model = "nl_core_news_lg"
  )

  score_responses(
    response_data = responses,
    possible_item_scores = 0:1,
    preprocessing = FALSE,
    spellcheck = FALSE,
    min_obs_accept = 1,
    min_certainty_accept = 0.5,
    spacy_model = "en_core_web_md"
  )

  expect_equal(captured_models, c("nl_core_news_lg", "en_core_web_md"))
  expect_equal(parse_count, 2L)
})
