# TODOs and Ideas

- [ ] Experiment with extractive QA (see the [notebook](notebooks/test_snippet_extraction.ipynb), lower half)
  - Snippets from the abstract or the title?
  - Title: binary classification (include as a snippet or not; as it is only one sentence)
  - Idea: Use QA models and measure overlap/spans that are both in the QA model output and the title/abstract (only first answer for title, top-k answers for abstract)
  - Fine-tuning possible with BioASQ training data
  - Choose a fitting pre-trained QA model from HuggingFace Hub (e.g., trained for medical QA)
- [ ] State transitions:
  - [ ] To documents, with conditions:
    - [ ] A
    - [ ] B
    - [ ] C
    - [ ] D
    - [ ] E
    - [ ] F
    - [ ] G
    - [ ] H
    - [ ] I
    - [ ] J
    - [ ] K
    - [ ] L
    - [ ] M
    - [ ] N
    - [ ] O
    - [ ] P
  - [ ] To snippets, with conditions:
    - [ ] A
    - [ ] B
    - [ ] C
    - [ ] D
    - [ ] E
    - [ ] F
    - [ ] G
    - [ ] H
    - [ ] I
    - [ ] J
    - [ ] K
    - [ ] L
    - [ ] M
    - [ ] N
    - [ ] O
    - [ ] P
  - [ ] To exact answer, with conditions:
    - [ ] A
    - [ ] B
    - [ ] C
    - [ ] D
    - [ ] E
    - [ ] F
    - [ ] G
    - [ ] H
    - [ ] I
    - [ ] J
    - [ ] K
    - [ ] L
    - [ ] M
    - [ ] N
    - [ ] O
    - [ ] P
  - [ ] To ideal answer, with conditions:
    - [ ] A
    - [ ] B
    - [ ] C
    - [ ] D
    - [ ] E
    - [ ] F
    - [ ] G
    - [ ] H
    - [ ] I
    - [ ] J
    - [ ] K
    - [ ] L
    - [ ] M
    - [ ] N
    - [ ] O
    - [ ] P

## Conditions for transitions

|  #  | Has documents | Has snippets | Has exact answer | Has ideal answer |
| :-: | :------------ | :----------- | :--------------- | :--------------- |
|  A  | no            | no           | no               | no               |
|  B  | no            | no           | no               | yes              |
|  C  | no            | no           | yes              | no               |
|  D  | no            | no           | yes              | yes              |
|  E  | no            | yes          | no               | no               |
|  F  | no            | yes          | no               | yes              |
|  G  | no            | yes          | yes              | no               |
|  H  | no            | yes          | yes              | yes              |
|  I  | yes           | no           | no               | no               |
|  J  | yes           | no           | no               | yes              |
|  K  | yes           | no           | yes              | no               |
|  L  | yes           | no           | yes              | yes              |
|  M  | yes           | yes          | no               | no               |
|  N  | yes           | yes          | no               | yes              |
|  O  | yes           | yes          | yes              | no               |
|  P  | yes           | yes          | yes              | yes              |
