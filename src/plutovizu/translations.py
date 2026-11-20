"""Traductions des textes écrits par plutovizu (légendes, axes, annotations), utilisées par
Chart.translate(lang). Le package écrit en français ; chaque entrée donne (en, es, ar).
None : pas de traduction, le texte reste tel quel."""
import re
from functools import lru_cache
from typing import Dict, Optional, Tuple

LANGS = ('fr', 'en', 'es', 'ar')

# Expressions remplacées partout où elles apparaissent (mot entier, la plus longue d'abord).
PHRASES: Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]] = {
    # Climat
    'Précipitations': ('Precipitation', 'Precipitación', 'الهطول'),
    'Température': ('Temperature', 'Temperatura', 'درجة الحرارة'),
    'Période sèche': ('Dry period', 'Período seco', 'الفترة الجافة'),
    'Période humide': ('Wet period', 'Período húmedo', 'الفترة الرطبة'),
    'seuil de sécheresse': ('drought threshold', 'umbral de sequía', 'عتبة الجفاف'),
    'Saison sèche': ('Dry season', 'Estación seca', 'الموسم الجاف'),
    'Saison humide': ('Wet season', 'Estación húmeda', 'الموسم الرطب'),
    'Fin de saison des pluies': ('End of rainy season', 'Fin de la estación lluviosa',
                                 'نهاية موسم الأمطار'),
    'Fin de saison sèche': ('End of dry season', 'Fin de la estación seca', 'نهاية الموسم الجاف'),
    'Pluie': ('Rain', 'Lluvia', 'المطر'),
    'Excédent': ('Surplus', 'Excedente', 'الفائض'),
    'Déficit': ('Deficit', 'Déficit', 'العجز'),
    'Évapotranspiration': ('Evapotranspiration', 'Evapotranspiración', 'التبخر والنتح'),
    'ETP': ('PET', 'ETP', 'ت.ن'),
    'Réserve utile': ('Available water capacity', 'Agua útil', 'الماء المتاح'),
    'Humidité relative': ('Relative humidity', 'Humedad relativa', 'الرطوبة النسبية'),
    'Humidité': ('Humidity', 'Humedad', 'الرطوبة'),
    'Zone de tolérance': ('Tolerance zone', 'Zona de tolerancia', 'منطقة التحمل'),
    'Zone optimale': ('Optimal zone', 'Zona óptima', 'المنطقة المثلى'),
    'Degrés-jours cumulés': ('Cumulative degree-days', 'Grados-día acumulados',
                             'الدرجات اليومية التراكمية'),
    'Écart à la référence': ('Departure from reference', 'Desviación respecto a la referencia',
                             'الانحراف عن المرجع'),
    'Moyenne glissante': ('Moving average', 'Media móvil', 'المتوسط المتحرك'),
    'minimum moyen du mois le plus froid': ('mean minimum of the coldest month',
                                            'mínima media del mes más frío',
                                            'متوسط الحد الأدنى لأبرد شهر'),
    # Étages bioclimatiques (Emberger)
    'Saharien': ('Saharan', 'Sahariano', 'صحراوي'),
    'Semi-aride': ('Semi-arid', 'Semiárido', 'شبه جاف'),
    'Aride': ('Arid', 'Árido', 'جاف'),
    'Subhumide': ('Sub-humid', 'Subhúmedo', 'شبه رطب'),
    'Perhumide': ('Perhumid', 'Perhúmedo', 'شديد الرطوبة'),
    'Humide': ('Humid', 'Húmedo', 'رطب'),
    'très froid': ('very cold', 'muy frío', 'بارد جدا'),
    'très chaud': ('very warm', 'muy cálido', 'حار جدا'),
    'froid': ('cold', 'frío', 'بارد'),
    'frais': ('cool', 'fresco', 'منعش'),
    'tempéré': ('temperate', 'templado', 'معتدل'),
    'chaud': ('warm', 'cálido', 'حار'),
    'sèche': ('dry', 'seca', 'جافة'),
    # Sols
    'Loam limono-argileux': ('Silty clay loam', 'Franco arcillo-limoso', 'طفال طيني طمي'),
    'Loam sablo-argileux': ('Sandy clay loam', 'Franco arcillo-arenoso', 'طفال طيني رملي'),
    'Loam argileux': ('Clay loam', 'Franco arcilloso', 'طفال طيني'),
    'Loam limoneux': ('Silt loam', 'Franco limoso', 'طفال طمي'),
    'Loam sableux': ('Sandy loam', 'Franco arenoso', 'طفال رملي'),
    'Sable loameux': ('Loamy sand', 'Arena franca', 'رمل طفالي'),
    'Argile sableuse': ('Sandy clay', 'Arcilla arenosa', 'طين رملي'),
    'Argile limoneuse': ('Silty clay', 'Arcilla limosa', 'طين طمي'),
    'Loam': ('Loam', 'Franco', 'طفال'),
    'Argile': ('Clay', 'Arcilla', 'الطين'),
    'Limon': ('Silt', 'Limo', 'الطمي'),
    'Sable': ('Sand', 'Arena', 'الرمل'),
    'Graviers': ('Gravel', 'Grava', 'الحصى'),
    'Diamètre des particules': ('Particle diameter', 'Diámetro de partícula', 'قطر الجسيمات'),
    'Passants cumulés': ('Cumulative passing', 'Pasante acumulado', 'النسبة المارة التراكمية'),
    'Capacité au champ': ('Field capacity', 'Capacidad de campo', 'السعة الحقلية'),
    'Point de flétrissement': ('Wilting point', 'Punto de marchitez', 'نقطة الذبول'),
    'Succion': ('Suction', 'Succión', 'الامتصاص'),
    'Profondeur': ('Depth', 'Profundidad', 'العمق'),
    'Vitesse mesurée': ('Measured rate', 'Velocidad medida', 'السرعة المقاسة'),
    'Vitesse modélisée': ('Modelled rate', 'Velocidad modelada', 'السرعة المنمذجة'),
    'Vitesse': ('Speed', 'Velocidad', 'السرعة'),
    'Cumul mesuré': ('Measured cumulative', 'Acumulado medido', 'التراكم المقاس'),
    'Modèle de': ('Model of', 'Modelo de', 'نموذج'),
    'Modèle': ('Model', 'Modelo', 'النموذج'),
    # Écologie et statistiques
    'Abondance relative': ('Relative abundance', 'Abundancia relativa', 'الوفرة النسبية'),
    "Rang de l'espèce": ('Species rank', 'Rango de la especie', 'رتبة النوع'),
    'Indice de Shannon': ('Shannon index', 'Índice de Shannon', 'مؤشر شانون'),
    "Nombre d'espèces": ('Number of species', 'Número de especies', 'عدد الأنواع'),
    'Nombre de sites': ('Number of sites', 'Número de sitios', 'عدد المواقع'),
    'Richesse cumulée': ('Cumulative richness', 'Riqueza acumulada', 'الثراء التراكمي'),
    'espèces': ('species', 'especies', 'نوعا'),
    'écart-type': ('standard deviation', 'desviación estándar', 'الانحراف المعياري'),
    'Corrélation': ('Correlation', 'Correlación', 'الارتباط'),
    'Inertie contrainte': ('Constrained inertia', 'Inercia restringida', 'القصور الذاتي المقيد'),
    'Effectif': ('Count', 'Frecuencia', 'العدد'),
    'Fréquence': ('Frequency', 'Frecuencia', 'التكرار'),
    'Observé': ('Observed', 'Observado', 'المرصود'),
    'observés': ('observed', 'observados', 'مرصودة'),
    'attendus': ('expected', 'esperados', 'متوقعة'),
    'Binomiale négative': ('Negative binomial', 'Binomial negativa', 'ذو الحدين السالب'),
    'Moyenne de': ('Mean of', 'Media de', 'متوسط'),
    'moyenne': ('mean', 'media', 'المتوسط'),
    'Variance': ('Variance', 'Varianza', 'التباين'),
    'Zéros': ('Zeros', 'Ceros', 'الأصفار'),
    'IC 95 %': ('95 % CI', 'IC 95 %', 'مجال ثقة 95 %'),
    'Ajustement': ('Fit', 'Ajuste', 'المطابقة'),
    'Observations': ('Observations', 'Observaciones', 'المشاهدات'),
    'Probabilité de présence': ('Probability of presence', 'Probabilidad de presencia',
                                'احتمال الوجود'),
    'Probabilité de': ('Probability of', 'Probabilidad de', 'احتمال'),
    'Réponse lissée': ('Smoothed response', 'Respuesta suavizada', 'الاستجابة الممهدة'),
    'Optimum': ('Optimum', 'Óptimo', 'الأمثل'),
    'Quantiles observés': ('Observed quantiles', 'Cuantiles observados', 'الكميات المرصودة'),
    'Quantiles théoriques': ('Theoretical quantiles', 'Cuantiles teóricos', 'الكميات النظرية'),
    'loi normale': ('normal distribution', 'distribución normal', 'التوزيع الطبيعي'),
    'Proportion': ('Proportion', 'Proporción', 'النسبة'),
    'Distance': ('Distance', 'Distancia', 'المسافة'),
    'Occurrences': ('Occurrences', 'Ocurrencias', 'التكرارات'),
    'Valeur normalisée': ('Normalised value', 'Valor normalizado', 'القيمة المعيارية'),
    'Biais': ('Bias', 'Sesgo', 'الانحياز'),
    'Différence': ('Difference', 'Diferencia', 'الفرق'),
    'Hors limites': ('Out of control', 'Fuera de control', 'خارج الحدود'),
    'Pourcentage cumulé': ('Cumulative percentage', 'Porcentaje acumulado',
                           'النسبة المئوية التراكمية'),
    'Cumul': ('Cumulative', 'Acumulado', 'التراكم'),
    'Seuil': ('Threshold', 'Umbral', 'العتبة'),
    'Essentielles': ('Vital few', 'Esenciales', 'الأساسية'),
    'Secondaires': ('Trivial many', 'Secundarias', 'الثانوية'),
    'Part de chaque': ('Share of each', 'Proporción de cada', 'حصة كل'),
    'largeur : part du total': ('width: share of total', 'ancho: proporción del total',
                                'العرض: الحصة من المجموع'),
    'et': ('and', 'y', 'و'),
    # Projets, réseaux, dates
    'Jours depuis le': ('Days since', 'Días desde el', 'أيام منذ'),
    'Semaine': ('Week', 'Semana', 'الأسبوع'),
    'Tâche critique': ('Critical task', 'Tarea crítica', 'مهمة حرجة'),
    'Tâche non critique': ('Non-critical task', 'Tarea no crítica', 'مهمة غير حرجة'),
    'Marge totale': ('Total float', 'Holgura total', 'الهامش الكلي'),
    'Temps': ('Time', 'Tiempo', 'الوقت'),
    'Date': ('Date', 'Fecha', 'التاريخ'),
    'Flot maximal': ('Maximum flow', 'Flujo máximo', 'التدفق الأقصى'),
    'Flux': ('Flow', 'Flujo', 'التدفق'),
    'capacité': ('capacity', 'capacidad', 'السعة'),
    'Lien saturé': ('Saturated link', 'Enlace saturado', 'رابط مشبع'),
    'Liens non retenus': ('Unused links', 'Enlaces descartados', 'روابط غير مختارة'),
    'Arbre couvrant': ('Spanning tree', 'Árbol de expansión', 'الشجرة الممتدة'),
    'Plus court chemin': ('Shortest path', 'Camino más corto', 'أقصر مسار'),
    'Tournée': ('Tour', 'Recorrido', 'الجولة'),
    'Départ': ('Start', 'Salida', 'الانطلاق'),
    'Vers la droite (au-dessus)': ('To the right (above)', 'Hacia la derecha (arriba)',
                                   'نحو اليمين (في الأعلى)'),
    'Vers la gauche (en dessous)': ('To the left (below)', 'Hacia la izquierda (abajo)',
                                    'نحو اليسار (في الأسفل)'),
    'Latitude': ('Latitude', 'Latitud', 'خط العرض'),
    'Longitude': ('Longitude', 'Longitud', 'خط الطول'),
}

# Libellés courts et ambigus (« Mar » : mardi ou mars ?) : traduits seulement quand le texte
# entier correspond, et seulement pour le graphique qui les écrit.
BY_CHART: Dict[str, Dict[str, Tuple[Optional[str], Optional[str], Optional[str]]]] = {
    'calendar_heatmap': {
        'Lun': ('Mon', 'Lun', 'الإثنين'), 'Mar': ('Tue', 'Mar', 'الثلاثاء'),
        'Mer': ('Wed', 'Mié', 'الأربعاء'), 'Jeu': ('Thu', 'Jue', 'الخميس'),
        'Ven': ('Fri', 'Vie', 'الجمعة'), 'Sam': ('Sat', 'Sáb', 'السبت'),
        'Dim': ('Sun', 'Dom', 'الأحد'),
    },
    'wind_rose': {
        'N': ('N', 'N', 'ش'), 'NE': ('NE', 'NE', 'ش.ق'), 'E': ('E', 'E', 'ق'),
        'SE': ('SE', 'SE', 'ج.ق'), 'S': ('S', 'S', 'ج'), 'SO': ('SW', 'SO', 'ج.غ'),
        'O': ('W', 'O', 'غ'), 'NO': ('NW', 'NO', 'ش.غ'),
    },
}


@lru_cache(maxsize=None)
def _table(lang: str):
    """Expressions → traduction pour une langue, et l'expression régulière qui les trouve."""
    i = LANGS.index(lang) - 1
    table = {}
    for fr, tr in PHRASES.items():
        if tr[i] is None:
            continue
        table[fr] = tr[i]
        if not fr[:2].isupper():                     # « Moyenne » comme « moyenne »
            flip = str.upper if fr[0].islower() else str.lower
            table.setdefault(flip(fr[0]) + fr[1:], flip(tr[i][0]) + tr[i][1:])
    keys = sorted(table, key=len, reverse=True)
    pattern = re.compile(r'(?<!\w)(' + '|'.join(re.escape(k).replace(r'\ ', r'\s+') for k in keys)
                         + r')(?!\w)')
    return table, pattern


def builtin(text: str, lang: str, chart_type: Optional[str] = None) -> str:
    """Traduit un texte écrit par plutovizu avec le dictionnaire intégré."""
    if lang == 'fr' or not text:
        return text
    if lang not in LANGS:
        raise ValueError(f"Langue '{lang}' sans traduction intégrée ({', '.join(LANGS)}) ; "
                         "from_='online' traduit vers toutes les langues")
    exact = BY_CHART.get(chart_type, {}).get(text.strip())
    if exact:
        return exact[LANGS.index(lang) - 1] or text
    table, pattern = _table(lang)

    def sub(m):
        found = m.group(1)
        tr = table[re.sub(r'\s+', ' ', found)]
        return tr.replace(' ', '\n', 1) if '\n' in found else tr   # garde le retour à la ligne

    return pattern.sub(sub, text)


def is_known(text: str, lang: str, chart_type: Optional[str] = None) -> bool:
    """Le dictionnaire intégré couvre-t-il tout le texte ? (hors expressions connues, il ne
    reste ni mot ni abréviation de deux lettres ou plus : « Précipitations (P) », « Jours
    depuis le 2024-06-01 »…)"""
    if lang not in LANGS[1:]:
        return False
    if text.strip() in BY_CHART.get(chart_type, {}):
        return True
    rest = _table(lang)[1].sub('', text)
    return rest != text and not re.search(r'[^\W\d_]{2,}', rest)


@lru_cache(maxsize=None)
def online(text: str, lang: str, source: str = 'fr') -> str:
    """Traduit un texte avec le package translate (service MyMemory, connexion requise).
    Mis en cache : un même texte n'est demandé qu'une fois par session."""
    try:
        from translate import Translator
    except ImportError as e:
        raise ImportError("from_='online' demande le package translate : "
                          "pip install translate (ou pip install plutovizu[translate])") from e
    try:
        out = Translator(to_lang=lang, from_lang=source).translate(text)
    except Exception as e:      # réseau, quota, langue inconnue…
        raise RuntimeError(f'Traduction en ligne impossible ({e})') from e
    if out.upper().startswith('MYMEMORY WARNING'):      # quota quotidien gratuit épuisé
        raise RuntimeError(f'Traduction en ligne impossible : {out}')
    return out
