# Extract from the title and abstract ONLY directly relevant sentences or phrases relevant to the given question. If no such sentences or phrases found, return an empty list.
SNIPPET_TEMPLATE = """
Here are 3 examples:
[Title]: Rethinking ramoplanin: the role of substrate binding in inhibition of peptidoglycan biosynthesis
[Abstract]: Ramoplanin is a cyclicdepsipeptide antibiotic that inhibits peptidoglycan biosynthesis. It was proposed in 1990 to block the MurG step of peptidoglycan synthesis by binding to the substrate of MurG, Lipid I. The proposed mechanism of MurG inhibition has become widely accepted even though it was never directly tested. In this paper, we disprove the accepted mechanism for how ramoplanin functions, and we present an alternative mechanism. This work has implications for the design of ramoplanin derivatives and may influence how other proposed substrate binding antibiotics are studied.
[Question]: Which was the first adeno-associated virus vector gene therapy product approved in the United States?
[Extracted]:
Title sentences: [empty list] (no sentences or phrases that directly answer the question)
Abstract sentences: [empty list] (no sentences or phrases that directly answer the question)
******************
[Title]: Rethinking ramoplanin: the role of substrate binding in inhibition of peptidoglycan biosynthesis
[Abstract]: Ramoplanin is a cyclicdepsipeptide antibiotic that inhibits peptidoglycan biosynthesis. It was proposed in 1990 to block the MurG step of peptidoglycan synthesis by binding to the substrate of MurG, Lipid I. The proposed mechanism of MurG inhibition has become widely accepted even though it was never directly tested. In this paper, we disprove the accepted mechanism for how ramoplanin functions, and we present an alternative mechanism. This work has implications for the design of ramoplanin derivatives and may influence how other proposed substrate binding antibiotics are studied.
[Question]: Which antibiotics target peptidoglycan biosynthesis?
[Extracted]: 
Title sentences: ["Rethinking ramoplanin: the role of substrate binding in inhibition of peptidoglycan biosynthesis."]
Abstract sentences: ["Ramoplanin is a cyclicdepsipeptide antibiotic that inhibits peptidoglycan biosynthesis."]
******************
[Title]: Mycobacterium Avium Complex (MAC) Lung Disease in Two Inner City Community Hospitals: Recognition, Prevalence, Co-Infection with Mycobacterium Tuberculosis (MTB) and Pulmonary Function (PF) Improvements After Treatment.
[Abstract]: Over 4 years, we evaluated patients who had positive MAC cultures, MAC infection and coinfection with MTB. Lung disease was related/likely related to MAC in 21 patients (50%) and not related in 21 (50%). In patients with MAC-related lung disease, the primary physician did not consider the diagnosis except when that physician was a pulmonologist. Half of those with MAC-related lung disease were smokers, white and US-born. There were 12 immunocompetent patients with MTB and NTM cultures. Eleven were non-white and all were foreign-born. Presentation and clinical course were consistent with MTB. All 8 patients with abnormal PF improved. The prevalence of MAC lung infection in two inner city hospitals was four times higher than that of TB. The indication for treatment of MAC infection should also rely heavily on clinical and radiological evidence when there is only one positive sputum culture. The diagnosis was considered only when the admitting physician was a pulmonologist. Most patients with combined infection were clinically consistent with MTB and responded to anti MTB treatment alone. Treatment with anti-MAC therapy improved PF in those patients whose PF was abnormal to begin with.
[Question]: Is Mycobacterium avium less susceptible to antibiotics than Mycobacterium tuberculosis?
[Extracted]:
Title sentences: [empty list] (no sentences or phrases that directly answer the question) 
Abstract sentences: ["The prevalence of MAC lung infection in two inner city hospitals was four times higher than that of TB.", "Most patients with combined infection were clinically consistent with MTB and responded to anti MTB treatment alone."]
******************


Here is the data:

[Title]: {title} 
[Abstract]: {abstract}
[Question]: {question}
[Extracted]:

"""

ANSWER_TEMPLATE_EXACT_YESNO = """
[INST]You are an accurate and reliable medical AI assistant that can answer questions with the help of external documents. It is very important that you answer the question as only yes or no.
Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate the correct answer based on your own knowledge. Answer "yes" or "no". You must not provide any additional comments

Here is an example:

Question: Is the protein Papilin secreted?

Document: The TSR superfamily is a diverse family of extracellular matrix and transmembrane proteins, many of which have functions related to regulating matrix organization, cell-cell interactions and cell guidance. This review samples some of the contemporary literature regarding TSR superfamily members (e.g. F-spondin, UNC-5, ADAMTS, papilin, and TRAP) where specific functions are assigned to the TSR domains.

Answer: yes [/INST]
******************

Here is the data:

Question: {question}

Document: {document}

Answer: 
"""

ANSWER_TEMPLATE_EXACT_LIST = """
[INST]You are an accurate and reliable medical AI assistant that can answer questions with the help of external documents. It is very important that you answer the question as a List of lists with single entities, each inside list contains only one entity, maximum 5 inside lists, no synonyms.
Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate the correct answer based on your own knowledge. Answer as a List of lists with single entities, each inside list contains only one entity, maximum 5 inside lists, no synonyms. You must not provide any additional comments.

Here is an example:

Question: List signaling molecules (ligands) that interact with the receptor EGFR?

Document: the epidermal growth factor receptor (EGFR) ligands, such as epidermal growth factor (EGF) and amphiregulin (AREG). EGFR ligands epidermal growth factor (EGF), amphiregulin (AREG) and transforming growth factor alpha (TGF\u03b1). Among EGFR ligands, heparin-binding EGF-like growth factor, TGF-\u03b1 and Betacellulin (BTC) are produced in the tumor microenvironment of FDC-S at RNA level. Plasma amphiregulin (AR), epidermal growth factor (EGF), transforming growth factor-\u03b1, and heparin binding-EGF were assessed by ELISA in 45 chemorefractory mCRC patients

Answer: [["epidermal growth factor"], ["betacellulin"], ["epiregulin"], ["transforming growth factor-\u03b1"], ["amphiregulin"]] [/INST]
******************

Here is the data:

Question: {question}

Document: {document}

Answer: 
"""

ANSWER_TEMPLATE_IDEAL_YESNO = """
[INST]You are an accurate and reliable medical AI assistant that can answer questions with the help of external documents. It is very important that you answer the question in at most 1 sentence.
Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate the correct answer based on your own knowledge.

Here is an example:

Question: Is the protein Papilin secreted?

Document: The TSR superfamily is a diverse family of extracellular matrix and transmembrane proteins, many of which have functions related to regulating matrix organization, cell-cell interactions and cell guidance. This review samples some of the contemporary literature regarding TSR superfamily members (e.g. F-spondin, UNC-5, ADAMTS, papilin, and TRAP) where specific functions are assigned to the TSR domains.

Answer: Yes,  papilin is a secreted protein[/INST]

Question: {question}

Document: {document}

Answer: 
"""

# [INST]You are an accurate and reliable medical AI assistant that can answer questions with the help of external documents. It is very important that you limit the length of your answer to 3 sentences in language that is understandable for non-professionals.
# Please note that external documents may contain noisy or factually incorrect information. If the information in the document contains the correct answer, you will give an accurate answer. If the information in the document does not contain the answer, you will generate ""I can not answer the question because of the insufficient information in documents.' If there are inconsistencies with the facts in some of the documents, please generate the response ""There are factual errors in the provided documents. and provide the correct answer.[/INST]

# Question: What are normal flu symptoms?

# Document: Chest Pain, Breathing, High Fever, and More People typically recover after about a week or two without any lasting problems. But sometimes this illness can lead to serious complications that require emergency care. Every year more than 200,000 people in the U.S. wind up in the hospital because of the flu. Tens of thousands die. Infants, the elderly, and people with certain diseases or weakened immune systems are the most at risk. But a flu emergency can happen to anyone. So it's important to know the signs of trouble. Normal Flu Symptoms Different strains of the influenza virus cause the flu. You get it when you inhale the germ or pick it up on your hands and then touch your eyes, nose, or mouth. Symptoms usually show up 1 to 4 days later. The flu can be hard to tell from a cold. But it usually comes on faster and is more severe.

# Answer: Normal flu symptoms include: High fever Headache Tiredness (can be extreme) Cough Sore throat Runny or stuffy nose Body aches Although flu vaccines can prevent certain strains, there's not much you can do after you get sick.

# Question: {question}

# Document: {document}

# Answer:
