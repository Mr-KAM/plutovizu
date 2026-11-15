# Kit logo Plutovizu

Tous les fichiers sont dérivés du même dessin vectoriel. Les textes sont convertis en tracés : aucune police à installer.

## Quelle version utiliser ? (logo responsive)

Choisissez la version selon la place disponible, de la plus large à la plus petite.

| Largeur disponible | Version | Dossier |
|---|---|---|
| 500 px et plus | Horizontal avec signature | `02-logo-horizontal-signature/` |
| 200 à 500 px | Horizontal | `01-logo-horizontal/` |
| Format carré ou étroit | Vertical | `03-logo-vertical/` |
| 48 à 200 px | Icône seule | `04-icone/` |
| 32 à 48 px | Favicon simplifié (3 barres) | `06-favicon-et-app/favicon.svg` |
| 16 px | Favicon micro (2 barres) | `06-favicon-et-app/favicon-micro.svg` |

Le nom seul (`05-nom-seul/`) sert quand l'icône est déjà présente à côté.

## Variantes de couleur

Chaque version existe en quatre variantes, indiquées dans le nom du fichier.

| Suffixe | À poser sur |
|---|---|
| `fond-clair` | Fonds blancs ou clairs |
| `fond-sombre` | Fonds sombres (mode sombre, bleu nuit) |
| `mono-noir` | Impression une couleur, gravure, tampon, fax |
| `mono-blanc` | Une couleur sur fond sombre ou sur photo |

Les SVG et PNG de ces dossiers ont un fond transparent. Les avatars de `04-icone/avec-fond/` et les fichiers de `06` et `08` ont un fond plein.

## Formats

- **SVG** : à préférer partout où c'est possible (web, README, documentation, Sphinx/MkDocs). Net à toute taille.
- **PNG** : pour les endroits qui refusent le SVG (PyPI, réseaux sociaux, diapositives). La largeur en pixels est dans le nom du fichier.
- **ICO** : `favicon.ico` contient les tailles 16, 32 et 48 px.
- **PDF** : vectoriel, pour l'impression (`07-impression-pdf/`).

## Usages courants

| Usage | Fichier |
|---|---|
| En-tête du README GitHub | `01-logo-horizontal/svg/plutovizu-horizontal-fond-clair.svg` et `-fond-sombre.svg` |
| Bannière README avec fond | `08-web-et-depot/banniere-readme-clair-1200x300.png` |
| Aperçu social du dépôt GitHub | `08-web-et-depot/github-social-preview-1280x640.png` |
| Aperçu de lien (Open Graph) | `08-web-et-depot/open-graph-1200x630.png` |
| Avatar GitHub / organisation / PyPI | `04-icone/avec-fond/plutovizu-avatar-fond-bleu-nuit-512px.png` |
| Logo de la documentation (Sphinx, MkDocs) | `04-icone/svg/` ou `01-logo-horizontal/svg/` |
| Favicon du site de documentation | `06-favicon-et-app/` |
| Diapositives, posters | `01` ou `03` en PNG 1600/1024 px, ou PDF |

### README qui suit le thème clair/sombre de GitHub

```html
<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo/plutovizu-horizontal-fond-sombre.svg">
    <img src="docs/logo/plutovizu-horizontal-fond-clair.svg" alt="Plutovizu" width="420">
  </picture>
</p>
```

### Favicon d'un site

```html
<link rel="icon" href="/favicon.ico" sizes="48x48">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/apple-touch-icon-180x180.png">
```

Les deux fichiers `android-chrome-*.png` sont prévus pour un `manifest.webmanifest` (icônes « maskable », marge de sécurité incluse).

## Couleurs

| Rôle | Hex |
|---|---|
| Bleu nuit (texte, orbite, fond sombre) | `#14123A` |
| Violet (barres, « vizu ») | `#5B4BDB` |
| Violet clair (sur fond sombre) | `#8577F2` / `#A79BFF` |
| Orange (dernière barre, lune) | `#FF7A2F` sur clair, `#FF8A3D` sur sombre |
| Fond clair | `#F7F6FF` |
| Disque de la planète | `#E4E0FF` sur clair, `#2B2670` sur sombre |

## Typographie

Space Grotesk (Google Fonts, licence SIL Open Font License) : Bold pour le nom, Medium pour la signature.

## Règles d'usage

- **Zone de protection** : laisser autour du logo un espace vide au moins égal au diamètre de la lune orange. Les fichiers des dossiers 01 à 05 incluent déjà cette marge.
- **Tailles minimales** : horizontal 120 px de large, vertical 80 px, icône complète 48 px. En dessous, passer au favicon simplifié.
- Ne pas déformer, recolorer, incliner, ni ajouter d'ombre ou de contour.
- Sur une photo ou un fond coloré, utiliser une version mono.
