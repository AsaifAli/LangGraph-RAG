from shared.source_policy import classify_source_intent


def test_external_benchmark_for_an_uploaded_quote_uses_both_sources():
    assert classify_source_intent(
        "Compare this uploaded supplier quotation against current market benchmarks."
    ) == "both"


def test_explicit_web_search_does_not_fall_back_to_an_unrelated_document():
    assert classify_source_intent(
        "Search the web for the latest public safety standards."
    ) == "web"


def test_explicit_document_summary_stays_document_grounded():
    assert classify_source_intent(
        "Summarize the uploaded supplier quotation."
    ) == "documents"


def test_general_question_remains_with_model_router():
    assert classify_source_intent("Explain retrieval-augmented generation.") == "auto"


def test_supplier_quote_market_fairness_question_uses_both_sources():
    assert classify_source_intent(
        "Compare this supplier quotation with the going market rate and tell me if it is fair."
    ) == "both"


def test_supplier_quote_fairness_question_uses_independent_external_evidence():
    assert classify_source_intent(
        "Is this supplier quote fair, overpriced, or a good deal?"
    ) == "both"
