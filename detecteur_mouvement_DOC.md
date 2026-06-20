# 🌲 detecteur_mouvement.py — Documentation complète

Script Python qui analyse des vidéos de caméra piège / déclenchement sur mouvement,
détecte automatiquement les passages avec activité, et extrait les clips correspondants
sans réencodage (copie directe, qualité d'origine préservée).

---

## Sommaire

1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Structure des dossiers](#structure-des-dossiers)
4. [Utilisation](#utilisation)
5. [Options en ligne de commande](#options-en-ligne-de-commande)
6. [Les trois presets de sensibilité](#les-trois-presets-de-sensibilité)
7. [Comment fonctionne la détection](#comment-fonctionne-la-détection)
8. [Format des fichiers en sortie](#format-des-fichiers-en-sortie)
9. [Paramètres fins à ajuster](#paramètres-fins-à-ajuster)
10. [Cas typiques et recettes](#cas-typiques-et-recettes)
11. [Dépannage](#dépannage)
12. [Pistes d'amélioration futures](#pistes-damélioration-futures)

---

## Prérequis

### Python
Version **3.10 ou supérieure** (utilise la syntaxe `list[tuple[float, float]]` et le
walrus operator `while chunk := ...`).

```bash
python3 --version
# doit afficher Python 3.10.x ou plus
```

### Bibliothèques Python

| Bibliothèque | Rôle | Installation |
|---|---|---|
| `opencv-python` | Lecture vidéo, détection de mouvement | `pip install opencv-python` |

> **Sur un VPS ou serveur sans interface graphique**, utilise la variante headless
> (plus légère, pas de dépendance à un système de fenêtres) :
> `pip install opencv-python-headless`
> En headless, l'option `--apercu` ne fonctionnera pas (pas de fenêtre possible).

### FFmpeg

FFmpeg est appelé en sous-processus pour **extraire les segments** détectés.
Il doit être installé séparément et accessible dans le PATH.

| OS | Commande d'installation |
|---|---|
| macOS | `brew install ffmpeg` |
| Ubuntu/Debian | `sudo apt install ffmpeg` |
| Windows | Télécharger sur [ffmpeg.org](https://ffmpeg.org/download.html), extraire, ajouter le dossier `bin/` au PATH système |

Vérification :
```bash
ffmpeg -version
# doit afficher la version sans erreur
```

---

## Installation

```bash
# 1. Cloner / copier le script dans un dossier de travail
mkdir mon-projet && cd mon-projet
cp /chemin/vers/detecteur_mouvement.py .

# 2. (Recommandé) Créer un environnement virtuel
python3 -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows

# 3. Installer OpenCV
pip install opencv-python

# 4. Vérifier FFmpeg
ffmpeg -version
```

---

## Structure des dossiers

Par défaut, le script s'attend à cette organisation :

```
mon-projet/
├── detecteur_mouvement.py
├── videos/                  ← dossier source (modifiable avec --input)
│   ├── video1.mp4
│   ├── video2.avi
│   └── sous-dossier/        ← les sous-dossiers sont parcourus récursivement
│       └── video3.mov
└── extraits/                ← dossier de sortie (modifiable avec --output)
    ├── video1/
    │   ├── video1_extrait_01_0-00-02-0-00-15.mp4
    │   └── video1_extrait_02_0-00-41-0-00-58.mp4
    └── video2/
        └── video2_extrait_01_0-01-03-0-01-22.mp4
```

Les dossiers `videos/` et `extraits/` sont **créés automatiquement** s'ils n'existent pas.
Tu peux pointer vers n'importe quel chemin absolu ou relatif avec `--input` et `--output`.

### Formats vidéo supportés

`.mp4` `.avi` `.mov` `.mkv` `.mts` `.m4v` (insensible à la casse : `.MP4`, `.AVI`…)

---

## Utilisation

### Cas le plus simple

Poser les vidéos dans `./videos/` et lancer :

```bash
python detecteur_mouvement.py
```

### Avec un dossier personnalisé

```bash
python detecteur_mouvement.py --input /Volumes/DisqueDur/CameraForet
```

### Tester sans extraire (voir juste les timestamps)

Utile pour valider la sensibilité avant de lancer l'extraction complète :

```bash
python detecteur_mouvement.py --input /mes/videos --liste
```

### Pipeline complet avec tous les paramètres

```bash
python detecteur_mouvement.py \
  --input  /media/sdcard/DCIM \
  --output /home/user/animaux/2024 \
  --sensibilite haute \
  --liste
```

---

## Options en ligne de commande

| Option | Valeur par défaut | Description |
|---|---|---|
| `--input DOSSIER` | `./videos` | Dossier contenant les vidéos source. Parcouru récursivement. |
| `--output DOSSIER` | `./extraits` | Dossier où seront créés les clips extraits. |
| `--sensibilite` | `normale` | Preset de détection : `haute`, `normale` ou `basse`. |
| `--liste` | _(flag)_ | Affiche les timestamps détectés sans lancer l'extraction FFmpeg. |
| `--apercu` | _(flag)_ | Ouvre une fenêtre de prévisualisation pendant l'analyse (nécessite opencv-python non-headless). |

---

## Les trois presets de sensibilité

### `haute` — Petits animaux, insectes, oiseaux rapides

- Détecte des changements très faibles (quelques pixels qui bougent)
- Risque accru de fausses détections : vent dans les feuilles, variation de lumière,
  insectes passant devant l'objectif
- À utiliser si tu rates des passages d'animaux rapides ou de petite taille

### `normale` _(défaut)_ — Mammifères, oiseaux de taille moyenne

- Bon équilibre entre sensibilité et précision
- Filtre les petites variations (vent léger, fluctuations lumineuses mineures)
- Convient à la majorité des vidéos de caméra piège en forêt

### `basse` — Grands animaux, ignorer vent et feuilles

- Exige qu'une surface plus grande de l'image bouge pour déclencher
- Élimine efficacement les fausses détections dues au vent, branches, pluie
- Peut rater des petits animaux ou des passages rapides en bordure de cadre

---

## Comment fonctionne la détection

### Algorithme : MOG2 (Mixture of Gaussians v2)

Le script utilise `cv2.createBackgroundSubtractorMOG2`, un algorithme de **soustraction
de fond** intégré à OpenCV. Il apprend progressivement à quoi ressemble le fond statique
de la scène (arbres, sol, ciel) et détecte tout ce qui s'en écarte.

**Étapes pour chaque image analysée :**

```
Frame brute
    ↓
Conversion en niveaux de gris          (réduit le bruit de couleur)
    ↓
Application du soustracteur de fond    (compare au modèle de fond appris)
    → produit un masque binaire : blanc = mouvement, noir = fond
    ↓
Morphologie (ouverture elliptique)     (supprime les petits points isolés / bruit)
    ↓
Comptage des pixels blancs restants
    ↓
Si surf_min < pixels_blancs < surf_max → timestamp enregistré
```

### Échantillonnage

Pour accélérer le traitement, le script n'analyse **pas toutes les images** : il en
prend environ 8 par seconde (`fps / 8`), ce qui est largement suffisant pour détecter
le passage d'un animal tout en divisant le temps de traitement par 3 à 4.

### Fusion des segments

Après l'analyse, les timestamps de mouvement sont post-traités en 3 étapes :

1. **Groupement** : les timestamps séparés de moins de `gap_fusion_sec` sont fusionnés
   en un seul segment (évite de découper un animal qui s'arrête une seconde).

2. **Marges** : chaque segment est étendu de `marge_avant_sec` avant et `marge_apres_sec`
   après (pour ne pas couper l'arrivée ou le départ de l'animal).

3. **Fusion des chevauchements** : si deux segments se chevauchent après l'ajout des
   marges, ils sont fusionnés en un seul.

### Extraction

L'extraction utilise **FFmpeg en mode copie directe** (`-c copy`) : aucun réencodage,
la qualité d'origine est strictement préservée et l'extraction est quasi-instantanée
(quelques secondes par clip, quel que soit la durée).

---

## Format des fichiers en sortie

Un sous-dossier par vidéo source est créé dans `--output` :

```
extraits/
└── ma_video/
    ├── ma_video_extrait_01_0-00-02-0-00-15.mp4
    ├── ma_video_extrait_02_0-00-41-0-00-58.mp4
    └── ma_video_extrait_03_0-02-14-0-02-31.mp4
```

Le nom du fichier encode directement le timecode :
`{nom_original}_extrait_{numéro}_{debut}-{fin}.mp4`

où `debut` et `fin` sont au format `H-MM-SS` (les `:` remplacés par `-` pour la
compatibilité Windows/macOS).

---

## Paramètres fins à ajuster

Tous les paramètres sont regroupés dans le dictionnaire `PRESETS` en haut du script.
Tu peux modifier les valeurs directement sans toucher au reste du code.

```python
PRESETS = {
    "normale": {
        "seuil_pixel":     25,   # [0-255] seuil de différence par pixel
                                  # ↑ = moins sensible, ↓ = plus sensible

        "surface_min_pct": 0.15, # % minimum de l'image qui doit bouger
                                  # ↑ = ignore les petits mouvements (insectes)

        "surface_max_pct": 70,   # % maximum — au-delà = flash/bug ignoré
                                  # utile pour ignorer les changements d'exposition

        "flou_kernel":     21,   # taille du noyau de lissage (doit être impair)
                                  # ↑ = plus tolérant au bruit pixel

        "marge_avant_sec": 2.0,  # secondes conservées AVANT le début du mouvement
        "marge_apres_sec": 4.0,  # secondes conservées APRÈS la fin du mouvement

        "gap_fusion_sec":  3.0,  # fusionne deux détections séparées de moins de X sec
                                  # ↑ = moins de clips courts, ↓ = plus de précision

        "min_duree_sec":   1.0,  # durée minimale d'un segment pour être conservé
                                  # filtre les fausses détections très brèves
    },
}
```

**Exemple de preset personnalisé** — caméra thermique, gros gibier uniquement :

```python
PRESETS["gibier"] = {
    "seuil_pixel":     50,
    "surface_min_pct": 1.0,
    "surface_max_pct": 50,
    "flou_kernel":     31,
    "marge_avant_sec": 3.0,
    "marge_apres_sec": 6.0,
    "gap_fusion_sec":  8.0,
    "min_duree_sec":   2.0,
}
```

Puis l'utiliser :
```bash
# Ajouter "gibier" dans les choices du parser et appeler avec :
python detecteur_mouvement.py --sensibilite gibier
```

---

## Cas typiques et recettes

### Beaucoup de fausses détections (vent, feuilles, pluie)

```bash
# Passer en sensibilité basse
python detecteur_mouvement.py --sensibilite basse

# Ou ajuster manuellement dans PRESETS["basse"] :
# augmenter surface_min_pct (ex: 1.0)
# augmenter seuil_pixel (ex: 50)
# augmenter min_duree_sec (ex: 3.0)
```

### Rates des animaux rapides ou petits

```bash
python detecteur_mouvement.py --sensibilite haute

# Ou dans PRESETS["haute"] :
# baisser seuil_pixel (ex: 10)
# baisser surface_min_pct (ex: 0.02)
# baisser min_duree_sec (ex: 0.3)
```

### Clips trop courts (animal tronqué)

Dans `PRESETS`, augmenter :
- `marge_avant_sec` : plus de contexte avant le mouvement
- `marge_apres_sec` : plus de contexte après
- `gap_fusion_sec` : fusionne les segments proches (animal qui s'arrête)

### Voir ce que le script détecte sans créer de fichiers

```bash
python detecteur_mouvement.py --liste
# affiche : Segment 01 : 0:00:14 → 0:00:27 (0:00:13)
```

### Visualiser la détection en direct

```bash
python detecteur_mouvement.py --apercu
# Appuie sur Q pour passer à la vidéo suivante
# Ne fonctionne pas en mode headless (serveur sans écran)
```

### Traiter un seul fichier (sans créer de dossier `videos/`)

Le script traite un dossier, pas un fichier seul. Astuce simple :

```bash
mkdir /tmp/une_video
cp ma_video.mp4 /tmp/une_video/
python detecteur_mouvement.py --input /tmp/une_video --output /tmp/extraits
```

---

## Dépannage

### `ModuleNotFoundError: No module named 'cv2'`

OpenCV n'est pas installé dans l'environnement Python actif.

```bash
pip install opencv-python
# ou, si plusieurs versions de Python :
python3 -m pip install opencv-python
```

Si tu utilises un venv, vérifie qu'il est bien activé (`source venv/bin/activate`).

### `ffmpeg: command not found`

FFmpeg n'est pas dans le PATH. L'analyse se déroule normalement mais l'extraction
échoue. Installer FFmpeg (voir section Prérequis) et relancer.

### `Impossible d'ouvrir : fichier.mp4`

Causes possibles :
- Fichier corrompu ou incomplet (transfert interrompu depuis la SD card)
- Codec non supporté par OpenCV — essayer de ré-encoder avec FFmpeg :
  ```bash
  ffmpeg -i original.mp4 -c:v libx264 -c:a aac converti.mp4
  ```

### Aucun mouvement détecté sur une vidéo qui en a

1. Tester avec `--sensibilite haute`
2. Vérifier avec `--apercu` que la vidéo s'ouvre correctement
3. Si la caméra fait un zoom ou un panoramique au déclenchement, tout l'arrière-plan
   bouge → `surface_max_pct` filtre ça. Baisser cette valeur (ex: 90) pour ne pas
   filtrer les grands mouvements.

### L'analyse est très lente

- Normale sur des vidéos 4K ou longues sur CPU uniquement
- Le script analyse environ 8 images/seconde de vidéo par seconde de traitement réel
- Une vidéo de 30 minutes en 1080p prend ~5-8 minutes sur un CPU standard
- Astuce : les caméras pièges produisent généralement du 720p ou 1080p, rarement du 4K

### Les clips extraits commencent quelques secondes trop tôt/tard

Ajuster `marge_avant_sec` et `marge_apres_sec` dans le preset concerné.

---

## Pistes d'amélioration futures

- **Ajout d'un preset en ligne de commande** : permettre `--sensibilite custom` avec
  passage direct des paramètres (`--seuil 30 --surface-min 0.2` etc.)

- **Détection par zone** : ignorer une zone de l'image (ex: un coin avec des feuilles
  qui bougent constamment) en appliquant un masque statique

- **Classification IA** : après détection, passer les clips dans un modèle de
  classification (animal / pas animal) pour filtrer les fausses détections résiduelles.
  Un modèle MobileNet ou YOLO léger pourrait tourner en CPU sans trop ralentir.

- **Export CSV/JSON** : générer un fichier de log avec tous les timestamps pour
  intégration dans un autre workflow

- **Traitement parallèle** : si plusieurs vidéos à traiter, utiliser
  `multiprocessing.Pool` pour analyser plusieurs fichiers en parallèle (un par cœur CPU)

- **Mode surveillance temps réel** : surveiller un dossier avec `watchdog` et lancer
  l'analyse automatiquement à chaque nouveau fichier déposé (intégration avec une
  caméra qui envoie ses fichiers par FTP/SFTP par exemple)

---

## Dépendances résumées

```
Python >= 3.10
├── opencv-python >= 4.5          (pip install opencv-python)
│   └── ou opencv-python-headless  (serveur sans écran)
└── stdlib uniquement : os, subprocess, argparse, pathlib, datetime

Système :
└── ffmpeg >= 4.0                 (dans le PATH)
```

---

*Script généré et documenté en juin 2026.*
