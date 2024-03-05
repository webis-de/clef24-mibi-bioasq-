from loguru import logger
from utils.snippets import SnippetExtractorQA


model = "deutsche-telekom/bert-multi-english-german-squad2"
tokenizer = "deutsche-telekom/bert-multi-english-german-squad2"
# model = "bigwiz83/sapbert-from-pubmedbert-squad2"
# tokenizer = "bigwiz83/sapbert-from-pubmedbert-squad2"

extractorQA = SnippetExtractorQA(model, tokenizer)


question = (
    "What is the mechanism by which HIV-1-encoded Vif protein allows virus replication?"
)

title = "HIV-1 subtype variability in Vif derived from molecular clones affects APOBEC3G-mediated host restriction"

abstract = "Background: The host protein APOBEC3G (A3G) can limit HIV-1 replication. Its protective effect is overcome by the HIV-1 'viral infectivity factor' (Vif), which targets A3G for proteosomal degradation. Although Vif is considered to be essential for HIV-1 replication, the effect of Vif variability among commonly used HIV-1 molecular clones of different genetic backgrounds on viral infectiousness and pathogenesis has not been fully determined. Methods: We cloned the intact Vif coding regions of available molecular clones of different subtypes into expression vectors. ?vif full-length HIV-1 clonal variants were generated from corresponding subtype-specific full-length molecular clones. Replication-competent viruses were produced in 293T cells in the presence or absence of A3G, with Vif being supplied by the full-length HIV-1 clone or in trans. The extent of A3G-mediated restriction was then determined in a viral replication assay using a reporter cell line. Results and conclusions: In the absence of A3G, Vif subtype origin did not impact viral replication. In the presence of A3G the subtype origin of Vif had a differential effect on viral replication. Vif derived from a subtype C molecular clone was less effective at overcoming A3G-mediated inhibition than Vif derived from either subtype B or CRF02_AG molecular clones."

result = extractorQA.extract(question, title, abstract)

logger.info(result[0])
logger.info("")
logger.info(result[1])
