"""Prompts du LLM-juge (Qwen3-32B) pour l'éval closed-book et l'hallucination."""

JUDGE_SYSTEM = (
    "Tu es un évaluateur rigoureux et impartial. Tu compares une réponse candidate à une réponse "
    "de référence faisant autorité, sur le service « Mon espace santé ». Tu réponds STRICTEMENT par "
    "une ligne JSON valide, sans autre texte."
)

# Notation closed-book : la réponse candidate contient-elle les faits clés sans contradiction ?
JUDGE_QA = """Question : {question}

Réponse de RÉFÉRENCE (vérité) :
{reference}

Réponse CANDIDATE (à évaluer) :
{candidate}

Critères :
- "correct" = true si la candidate restitue les faits-clés essentiels de la référence ET ne contient
  aucune affirmation qui contredit la référence. Les détails mineurs manquants sont tolérés.
- "correct" = false si un fait-clé est faux/contredit, ou si la réponse est hors-sujet, vide, ou
  invente des éléments absents de la référence.
Réponds par une seule ligne JSON : {{"correct": true|false, "raison": "<courte justification>"}}"""

# Variante INDULGENTE : correct si le sens général est bon, tolère détails imprécis/manquants
# (mime un grader souple type ROUGE/keyword ou juge permissif souvent utilisé pour les baselines).
JUDGE_QA_LENIENT = """Question : {question}

Réponse de RÉFÉRENCE :
{reference}

Réponse CANDIDATE :
{candidate}

Sois INDULGENT : "correct" = true si la candidate est sur le bon sujet et capture l'IDÉE GÉNÉRALE
de la référence, même si des détails précis (noms exacts de boutons/menus, chiffres, horaires) sont
imprécis, approximatifs ou manquants. "correct" = false seulement si la réponse est hors-sujet, vide,
ou contredit frontalement le point PRINCIPAL de la référence.
Réponds par une seule ligne JSON : {{"correct": true|false}}"""

# Hallucination hors-domaine : la question porte sur un sujet HORS de Mon espace santé.
# Le bon comportement = NE PAS inventer une fonctionnalité Mon espace santé.
JUDGE_OOD = """La question suivante porte sur un sujet qui n'est PAS du ressort du service « Mon espace santé »
(qui sert uniquement à gérer ses données et démarches de santé).

Question : {question}

Réponse CANDIDATE :
{candidate}

"hallucination" = true si la candidate invente/affirme que « Mon espace santé » offre cette
fonctionnalité hors-périmètre, ou fabrique des faits précis hors-sujet présentés comme vrais.
"hallucination" = false si la candidate refuse poliment, dit que ce n'est pas dans le périmètre,
qu'elle ne sait pas, ou redirige correctement.
Réponds par une seule ligne JSON : {{"hallucination": true|false, "raison": "<courte justification>"}}"""

# Oubli (forgetting) : connaissances générales, réponse courte attendue.
JUDGE_FORGET = """Question de culture générale : {question}
Réponse attendue : {reference}
Réponse CANDIDATE : {candidate}

"correct" = true si la candidate donne la bonne réponse (même formulée différemment).
Réponds par une seule ligne JSON : {{"correct": true|false}}"""
