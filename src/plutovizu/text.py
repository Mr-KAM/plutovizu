"""Outils texte pour l'analyse qualitative : découpage en mots, mots vides, fréquences.

    from plutovizu import word_counts
    word_counts(df['reponse'], ngram=2).most_common(10)   # expressions de deux mots
"""
import re
from collections import Counter
from typing import Iterable, Optional, Set

# Mots vides français courants (articles, pronoms, prépositions, auxiliaires…)
FRENCH_STOPWORDS: Set[str] = set("""
a à afin ai aie aient ainsi alors as au aucun aucune aujourd auprès aussi autre autres aux avaient
avais avait avant avec avez aviez avions avoir avons ayant bien c ça car ce ceci cela celle celles
celui cependant certain certaines certains ces cet cette ceux chacun chaque chez ci comme comment d
dans de des depuis dès donc dont du durant elle elles en encore entre es est et étaient étais était
étant été être eu eux fait faire fais faut fois font hors ici il ils j je jusqu l la là laquelle le
lequel les lesquels leur leurs lors lui m ma mais me même mêmes mes moi moins mon n ne ni non nos
notre nous on ont ou où par parce pas peu peut peuvent plus plutôt pour pourquoi puis qu quand que
quel quelle quelles quels qui quoi s sa sans se selon ses si sien soi soit sommes son sont sous
souvent suis sur t ta tandis te tel telle telles tels tes toi ton tous tout toute toutes très tu un
une unes uns va vers via voici voilà vont vos votre vous y
""".split())

ENGLISH_STOPWORDS: Set[str] = set("""
a about above after again against all am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have
having he her here hers herself him himself his how i if in into is it its itself just me more most
my myself no nor not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these they this those
through to too under until up very was we were what when where which while who whom why will with
would you your yours yourself yourselves
""".split())

_WORD = re.compile(r"[^\W\d_]+(?:[-'][^\W\d_]+)*")  # lettres (accents compris), mots composés


def stopwords_for(language: Optional[str]) -> Set[str]:
    """Mots vides de la langue : 'fr', 'en', 'fr+en' ou None (aucun)."""
    table = {'fr': FRENCH_STOPWORDS, 'en': ENGLISH_STOPWORDS}
    if not language:
        return set()
    try:
        return set().union(*(table[lang] for lang in language.split('+')))
    except KeyError:
        raise ValueError(f"language doit valoir 'fr', 'en', 'fr+en' ou None, pas '{language}'") from None


def word_counts(texts: Iterable, language: Optional[str] = 'fr',
                stopwords: Optional[Iterable[str]] = None, min_length: int = 3,
                ngram: int = 1) -> Counter:
    """Fréquence des mots (ou des expressions de ``ngram`` mots) dans une série de textes.

    Les textes sont mis en minuscules ; les mots vides de ``language``, ceux de
    ``stopwords`` et les mots de moins de ``min_length`` lettres sont ignorés. Les
    expressions ne franchissent pas un mot vide (« prix de l'engrais » → « prix engrais »
    n'est pas formé).
    """
    if isinstance(texts, str):
        texts = [texts]
    ignore = stopwords_for(language) | {w.lower() for w in (stopwords or [])}
    counts = Counter()
    for text in texts:
        if not isinstance(text, str):
            continue
        run = []  # suite de mots utiles consécutifs
        for token in _WORD.findall(text.lower()) + [None]:
            token = token.split("'")[-1] if token else token  # l'eau → eau, d'azote → azote
            if token and token not in ignore and len(token) >= min_length:
                run.append(token)
                continue
            counts.update(' '.join(run[i:i + ngram]) for i in range(len(run) - ngram + 1))
            run = []
    return counts
