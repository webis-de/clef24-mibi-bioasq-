SNIPPET_PROMPT = """
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
