"""
Détecteur de mouvement pour vidéos de forêt (pièges photographiques)
=====================================================================
Extrait automatiquement les passages avec mouvement de tes vidéos.

Usage:
    python detecteur_mouvement.py                          # traite ./videos/, résultats dans ./extraits/
    python detecteur_mouvement.py --input /chemin/videos   # dossier source personnalisé
    python detecteur_mouvement.py --sensibilite haute      # plus sensible (petits animaux)
    python detecteur_mouvement.py --sensibilite basse      # moins sensible (ignore vent/feuilles)
    python detecteur_mouvement.py --apercu                 # prévisualise avant d'extraire

Formats supportés : .mp4, .avi, .mov, .mkv, .mts, .m4v
"""

import cv2
import os
import subprocess
import argparse
from pathlib import Path
from datetime import timedelta


# ─── Paramètres de détection ────────────────────────────────────────────────

PRESETS = {
    "haute": {
        # Très sensible — capte insectes, petits oiseaux, légères secousses
        "seuil_pixel":       15,   # différence de pixel pour "changement" (0-255)
        "surface_min_pct":   0.05, # % de l'image qui doit bouger (0.05 = 0.05%)
        "surface_max_pct":   80,   # au-delà = probablement un bug/flash de lumière
        "flou_kernel":       11,   # lissage avant comparaison (impair)
        "marge_avant_sec":   1.5,  # secondes ajoutées avant le début du mouvement
        "marge_apres_sec":   3.0,  # secondes ajoutées après la fin du mouvement
        "gap_fusion_sec":    2.0,  # fusionne segments séparés de moins de X secondes
        "min_duree_sec":     0.5,  # ignore détections inférieures à cette durée
    },
    "normale": {
        "seuil_pixel":       25,
        "surface_min_pct":   0.15,
        "surface_max_pct":   70,
        "flou_kernel":       21,
        "marge_avant_sec":   2.0,
        "marge_apres_sec":   4.0,
        "gap_fusion_sec":    3.0,
        "min_duree_sec":     1.0,
    },
    "basse": {
        # Ignore vent, feuilles qui bougent, variations lumineuses
        "seuil_pixel":       40,
        "surface_min_pct":   0.5,
        "surface_max_pct":   60,
        "flou_kernel":       31,
        "marge_avant_sec":   2.0,
        "marge_apres_sec":   5.0,
        "gap_fusion_sec":    5.0,
        "min_duree_sec":     2.0,
    },
}


# ─── Détection de mouvement ──────────────────────────────────────────────────

def detecter_segments(chemin_video: str, params: dict, apercu: bool = False) -> list[tuple[float, float]]:
    """
    Analyse une vidéo et retourne une liste de (debut_sec, fin_sec)
    correspondant aux passages avec mouvement.
    """
    cap = cv2.VideoCapture(chemin_video)
    if not cap.isOpened():
        print(f"  ✗ Impossible d'ouvrir : {chemin_video}")
        return []

    fps        = cap.get(cv2.CAP_PROP_FPS) or 25
    total_img  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    largeur    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duree_sec  = total_img / fps

    surface_totale = largeur * hauteur
    surf_min = surface_totale * params["surface_min_pct"] / 100
    surf_max = surface_totale * params["surface_max_pct"] / 100

    print(f"  Résolution : {largeur}×{hauteur} | {fps:.1f} fps | Durée : {_fmt(duree_sec)}")

    # On analyse 1 image sur 3 pour aller plus vite (suffisant pour la détection)
    echantillonnage = max(1, round(fps / 8))

    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=max(20, round(fps * 2)),
        varThreshold=params["seuil_pixel"],
        detectShadows=False,
    )
    noyau = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (params["flou_kernel"], params["flou_kernel"])
    )

    timestamps_mouvement = []  # secondes où du mouvement est détecté
    num_img = 0
    affichage_precedent = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if num_img % echantillonnage == 0:
            gris   = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            masque = bg_subtractor.apply(gris)
            masque = cv2.morphologyEx(masque, cv2.MORPH_OPEN, noyau)

            surf_mouvement = cv2.countNonZero(masque)

            if surf_min < surf_mouvement < surf_max:
                ts = num_img / fps
                timestamps_mouvement.append(ts)

                if apercu:
                    pct = int(surf_mouvement / surface_totale * 100)
                    cv2.putText(frame, f"MOUVEMENT {pct}%", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

            if apercu:
                cv2.imshow("Apercu - Q pour quitter", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    apercu = False
                    cv2.destroyAllWindows()

        # Barre de progression
        pct_traite = int(num_img / max(total_img, 1) * 50)
        if pct_traite != affichage_precedent:
            barre = "█" * pct_traite + "░" * (50 - pct_traite)
            print(f"\r  [{barre}] {pct_traite * 2}%", end="", flush=True)
            affichage_precedent = pct_traite

        num_img += 1

    print()  # saut de ligne après la barre
    cap.release()
    if apercu:
        cv2.destroyAllWindows()

    return _fusionner_segments(timestamps_mouvement, params, duree_sec)


def _fusionner_segments(timestamps: list[float], params: dict, duree_video: float) -> list[tuple[float, float]]:
    """
    Convertit une liste de timestamps en segments (debut, fin)
    en fusionnant les détections proches et en ajoutant des marges.
    """
    if not timestamps:
        return []

    gap      = params["gap_fusion_sec"]
    av       = params["marge_avant_sec"]
    ap       = params["marge_apres_sec"]
    min_dur  = params["min_duree_sec"]

    # 1. Fusionner les timestamps proches en groupes
    groupes = []
    debut = timestamps[0]
    fin   = timestamps[0]

    for ts in timestamps[1:]:
        if ts - fin <= gap:
            fin = ts
        else:
            groupes.append((debut, fin))
            debut = fin = ts
    groupes.append((debut, fin))

    # 2. Appliquer marges + filtrer les trop courts + borner à la durée vidéo
    segments = []
    for d, f in groupes:
        d_marge = max(0.0, d - av)
        f_marge = min(duree_video, f + ap)
        if (f_marge - d_marge) >= min_dur:
            segments.append((d_marge, f_marge))

    # 3. Fusionner les segments qui se chevauchent après l'ajout des marges
    fusionnes = []
    for seg in segments:
        if fusionnes and seg[0] <= fusionnes[-1][1]:
            fusionnes[-1] = (fusionnes[-1][0], max(fusionnes[-1][1], seg[1]))
        else:
            fusionnes.append(list(seg))

    return [(d, f) for d, f in fusionnes]


# ─── Extraction avec FFmpeg ──────────────────────────────────────────────────

def extraire_segments(chemin_video: str, segments: list, dossier_sortie: str, nom_base: str):
    """Extrait chaque segment de la vidéo avec FFmpeg (copie directe, sans réencodage)."""
    os.makedirs(dossier_sortie, exist_ok=True)
    extraits = []

    for i, (debut, fin) in enumerate(segments, 1):
        nom_fichier = f"{nom_base}_extrait_{i:02d}_{_fmt(debut, fs=True)}-{_fmt(fin, fs=True)}.mp4"
        chemin_sortie = os.path.join(dossier_sortie, nom_fichier)

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(debut),
            "-to", str(fin),
            "-i", chemin_video,
            "-c", "copy",          # copie directe — rapide, sans perte de qualité
            "-avoid_negative_ts", "make_zero",
            chemin_sortie
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            taille = os.path.getsize(chemin_sortie) / 1024 / 1024
            print(f"    ✓ Extrait {i:02d} : {_fmt(debut)} → {_fmt(fin)} ({taille:.1f} Mo)")
            extraits.append(chemin_sortie)
        else:
            print(f"    ✗ Erreur extrait {i} : {result.stderr[-200:]}")

    return extraits


# ─── Utilitaires ────────────────────────────────────────────────────────────

def _fmt(secondes: float, fs: bool = False) -> str:
    """Formate des secondes en HH:MM:SS ou HH-MM-SS (pour noms de fichiers)."""
    td = str(timedelta(seconds=int(secondes)))
    if td.count(':') == 1:
        td = "0:" + td  # ajoute les heures si manquantes
    return td.replace(":", "-") if fs else td


def trouver_videos(dossier: str) -> list[str]:
    extensions = {".mp4", ".avi", ".mov", ".mkv", ".mts", ".m4v", ".MP4", ".AVI", ".MOV"}
    return sorted([
        str(p) for p in Path(dossier).rglob("*")
        if p.suffix in extensions
    ])


# ─── Programme principal ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extrait automatiquement les passages avec mouvement de vidéos de forêt."
    )
    parser.add_argument("--input",       default="./videos",  help="Dossier contenant les vidéos (défaut: ./videos)")
    parser.add_argument("--output",      default="./extraits", help="Dossier de sortie (défaut: ./extraits)")
    parser.add_argument("--sensibilite", choices=["haute", "normale", "basse"], default="normale",
                        help="Sensibilité de détection (défaut: normale)")
    parser.add_argument("--apercu",      action="store_true", help="Afficher la vidéo pendant l'analyse")
    parser.add_argument("--liste",       action="store_true", help="Afficher les segments sans extraire")
    args = parser.parse_args()

    params = PRESETS[args.sensibilite]
    videos = trouver_videos(args.input)

    if not videos:
        print(f"\n⚠️  Aucune vidéo trouvée dans : {args.input}")
        print("   Crée un dossier 'videos/' à côté du script et place tes fichiers dedans.")
        return

    print(f"\n🌲 Détecteur de mouvement — {len(videos)} vidéo(s) trouvée(s)")
    print(f"   Sensibilité : {args.sensibilite} | Dossier sortie : {args.output}\n")

    total_extraits = 0
    total_videos_avec_mouvement = 0

    for i, chemin in enumerate(videos, 1):
        nom = Path(chemin).name
        print(f"[{i}/{len(videos)}] 📹 {nom}")

        segments = detecter_segments(chemin, params, apercu=args.apercu)

        if not segments:
            print("  → Aucun mouvement détecté\n")
            continue

        total_videos_avec_mouvement += 1
        duree_totale = sum(f - d for d, f in segments)
        print(f"  → {len(segments)} segment(s) détecté(s) | Durée totale : {_fmt(duree_totale)}")

        for j, (d, f) in enumerate(segments, 1):
            print(f"     Segment {j:02d} : {_fmt(d)} → {_fmt(f)} ({_fmt(f-d)})")

        if not args.liste:
            nom_base = Path(chemin).stem
            sous_dossier = os.path.join(args.output, nom_base)
            extraits = extraire_segments(chemin, segments, sous_dossier, nom_base)
            total_extraits += len(extraits)

        print()

    print("─" * 60)
    print(f"✅ Terminé ! {total_videos_avec_mouvement}/{len(videos)} vidéos avec mouvement")
    if not args.liste:
        print(f"   {total_extraits} extrait(s) sauvegardé(s) dans : {args.output}/")


if __name__ == "__main__":
    main()
