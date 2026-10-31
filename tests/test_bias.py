import pandas as pd
from src.bias.lexicon import swap_map, bias_terms
from src.bias.mitigation import swap_terms, contains_bias_term, counterfactual_augment

def test_bias_lexicon():
    swaps = swap_map()
    assert "እሱ" in swaps
    assert swaps["እሱ"] == "እሷ"
    assert swaps["እሷ"] == "እሱ"
    assert "ዶክተር" in swaps
    assert swaps["ዶክተር"] == "ነርስ"

def test_swap_terms():
    text = "እሱ ጎበዝ ዶክተር ነው።"
    swapped, count = swap_terms(text)
    assert count == 2
    assert "እሷ" in swapped
    assert "ነርስ" in swapped

def test_contains_bias_term():
    assert contains_bias_term("የእሱ መጽሐፍ ነው")
    assert not contains_bias_term("ዛሬ ዝናብ እየዘነበ ነው")

def test_counterfactual_augment():
    df = pd.DataFrame({
        "id": [f"sent-{i}" for i in range(10)],
        "text": ["እሱ ጥሩ ሰው ነው"] * 5 + ["ዛሬ ፀሐይ ወጥቷል"] * 5,
        "label": ["positive"] * 10
    })
    aug_df, stats = counterfactual_augment(df, target_expansion=0.3, seed=42)
    assert len(aug_df) >= len(df)
    assert stats["realised_expansion"] > 0
