import cv2 as cv
import numpy as np

def show_image(title, image):
    image = cv.resize(image, (0, 0), fx=0.2, fy=0.2)
    cv.imshow(title, image)
    cv.waitKey(0)
    # cv.destroyAllWindows()

padding = 25  # Padding pentru marginile celulelor

def extrage_careu(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # intervale de nuante de verde pentru extragerea careului
    lower_green = np.array([35, 30, 30])
    upper_green = np.array([85, 255, 255])

    # crearea mastii si dilatare
    mask = cv.inRange(hsv, lower_green, upper_green)
    mask = cv.dilate(mask, np.ones((85, 85), np.uint8), iterations=1)

    # punctele cu valoare > 0 sunt de la tabla de joc
    points = np.column_stack(np.where(mask > 0))
    pts = np.array([[p[1], p[0]] for p in points], dtype=np.int32)

    # determinam colturile cele mai indepartate (extreme)
    sums = pts[:, 0] + pts[:, 1]
    diffs = pts[:, 0] - pts[:, 1]

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]


    # dimensiunea careului extras la care adaugam si paddingul
    width, height = 1600 + 2 * padding, 1600 + 2 * padding

    source = np.array([top_left, bottom_left, bottom_right, top_right], dtype="float32")
    dest = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")

    M = cv.getPerspectiveTransform(source, dest)
    result = cv.warpPerspective(image, M, (width, height))

    return result


lines_horizontal = []
for i in range(padding, 1601 + padding, 100):
    l = []
    l.append((padding, i))
    l.append((1599 + padding, i))
    lines_horizontal.append(l)


lines_vertical = []
for i in range(padding, 1601 + padding, 100):
    l = []
    l.append((i, padding))
    l.append((i, 1599 + padding))
    lines_vertical.append(l)


def determina_configuratie_careu(careu, lines_horizontal, lines_vertical, templates_dict, template_images):

    matrix = np.empty((16, 16), dtype='<U2')  # Creează o matrice goală de dimensiune 16×16 cu elemente sting
    careu = cv.normalize(careu, None, 0, 255, cv.NORM_MINMAX)

    # parcurge fiecare celula
    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):
            y_min = lines_vertical[j][0][0]
            y_max = lines_vertical[j + 1][1][0]
            x_min = lines_horizontal[i][0][1]
            x_max = lines_horizontal[i + 1][1][1]

            # extrage celula cu un padding pentru un rezultat mai bun
            x1 = x_min - padding
            x2 = x_max + padding
            y1 = y_min - padding
            y2 = y_max + padding

            patch = careu[x1:x2, y1:y2].copy()

            best_score = 0
            best_label = "__"

            for fname, label in templates_dict.items():
                if fname == "bonus_1.png" or fname == "bonus_2.png":
                    continue
                template = template_images[fname]
                res = cv.matchTemplate(patch, template, cv.TM_CCOEFF_NORMED)

                min_val, max_val, _, _ = cv.minMaxLoc(res)
                if max_val > best_score:
                    best_score = max_val
                    best_label = label

            threshold = 0.65
            if best_score < threshold:
                best_label = "__"

            matrix[i, j] = best_label
    return matrix


def detecteaza_culori(matrix, careu_bgr):

    # intervalele HSV pentru fiecare culoare
    color_masks = {
        'W': [(np.array([0, 0, 100]), np.array([180, 60, 255]))],
        'R': [(np.array([0, 120, 50]), np.array([5, 255, 255])),
              (np.array([175, 120, 50]), np.array([180, 255, 255]))],
        'O': [(np.array([5, 70, 50]), np.array([20, 255, 255]))],
        'Y': [(np.array([26, 50, 50]), np.array([40, 255, 255]))],
        'G': [(np.array([41, 50, 50]), np.array([90, 255, 255]))],
        'B': [(np.array([91, 50, 50]), np.array([130, 255, 255]))]
    }

    for i in range(len(lines_horizontal) - 1):
        for j in range(len(lines_vertical) - 1):

            label = matrix[i, j]
            if label in ["__", "+1", "+2"]: # cautam culoarea doar pentru piese de joc
                continue

            y1 = lines_horizontal[i][0][1]
            y2 = lines_horizontal[i + 1][0][1]
            x1 = lines_vertical[j][0][0]
            x2 = lines_vertical[j + 1][0][0]

            patch = careu_bgr[y1:y2, x1:x2].copy()
            hsv = cv.cvtColor(patch, cv.COLOR_BGR2HSV)

            # testeaza fiecare culoare si gaseste cea mai buna potrivire
            best_color = 'W'
            best_score = 0

            for color_code, ranges in color_masks.items():
                total_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)

                for lower, upper in ranges:
                    mask = cv.inRange(hsv, lower, upper)
                    total_mask = cv.bitwise_or(total_mask, mask)

                score = np.sum(total_mask > 0) / total_mask.size

                if score > best_score:
                    best_score = score
                    best_color = color_code

            matrix[i, j] = f"{label[0]}{best_color}"

    return matrix


# adauga manual bonusul pentru un cadran in functie de pozitionare
def add_bonus(matrix, lin, col):
    if matrix[lin][col] == "__":
        matrix[lin][col] = "+2"
        matrix[lin + 5][col + 5] = "+2"

        for i in range(1, 6):
            matrix[lin + 5 - i, col + i - 1] = "+1"
            matrix[lin + 6 - i, col + i] = "+1"
    else:
        matrix[lin][col + 5] = "+2"
        matrix[lin + 5][col] = "+2"

        for i in range(1, 6):
            matrix[lin + i - 1, col + i] = "+1"
            matrix[lin + i, col + i - 1] = "+1"

    return matrix


# deschidem la culoare pixelii din padding, care nu apartin pieselor de joc, ca sa evitam detectiile false
def clean_padding(careu_gray, padding=25, thresh=100):
    h, w = careu_gray.shape
    out = careu_gray.copy()

    # sus
    top = out[:padding, :]
    mask = top > thresh
    top[mask] = 200

    # jos
    bottom = out[h-padding:h, :]
    mask = bottom > thresh
    bottom[mask] = 200

    # stanga
    left = out[:, :padding]
    mask = left > thresh
    left[mask] = 200

    # dreapta
    right = out[:, w-padding:w]
    mask = right > thresh
    right[mask] = 200

    return out



# extinde o piesa pe orizontala sau verticala
def extinde_directie(curent, start, dl, dc):
    n = 16
    lin0, col0 = start
    rezultat = []

    # dreapta sau jos
    l, c = lin0 + dl, col0 + dc
    # cat timp avem piese si nu s-a terminat careul
    while 0 <= l < n and 0 <= c < n and curent[l, c] not in ["__", "+1", "+2"]:
        rezultat.append((l, c))
        l += dl
        c += dc

    # stanga sau sus
    l, c = lin0 - dl, col0 - dc
    while 0 <= l < n and 0 <= c < n and curent[l, c] not in ["__", "+1", "+2"]:
        rezultat.append((l, c))
        l -= dl
        c -= dc

    return rezultat


# calculeaza scorul pe baza diferentei dintre cele doua matrici
def calculeaza_scor(anterior, curent):
    n = 16
    scor = 0

    piese_noi = set()
    piese_detectate = []
    for lin in range(n):
        for col in range(n):
            # daca inainte era gol sau bonus si acum e piesa
            if anterior[lin, col] in ["__", "+1", "+2"] and curent[lin, col] not in ["__", "+1", "+2"]:
                piese_noi.add((lin, col)) # adaugam coordonatele piesei
                # adaugam la lista pe care o vom da ca output
                piese_detectate.append(f"{lin + 1}{chr(ord('A') + col)} {curent[lin, col]}")


    linii_procesate = set()

    for (lin, col) in piese_noi:
        # pentru fiecare punct extindem pe verticala si orizontala
        for dl, dc, directie in [(0, 1, 'H'), (1, 0, 'V')]:
            vecini = extinde_directie(curent, (lin, col), dl, dc)
            linie = [(lin, col)] + vecini

            if len(linie) < 2:
                continue  # e doar o piesa deci nu luam in calcul

            # cheie pentru a identifica daca o linia a mai fost gasita odata
            # tinem minte directie, inceputul si sfarsitul
            if directie == 'H':
                cols = [p[1] for p in linie]
                (start_col, end_col) = min(cols), max(cols)
                key = ('H', lin, start_col, end_col)
            else:
                linii = [p[0] for p in linie]
                start_lin, end_lin = min(linii), max(linii)
                key = ('V', col, start_lin, end_lin)

            # daca am mai facut odata linia asta trecem peste
            if key in linii_procesate:
                continue

            linii_procesate.add(key)

            # scor linie: lungimea + qwirkle bonus (daca sunt 6)
            lungime = len(linie)
            scor_l = lungime
            if lungime == 6:
                scor_l += 6

            # bonusuri
            for (x, y) in linie:
                if (x, y) in piese_noi:
                    if anterior[x, y] == "+1":
                        scor_l += 1
                        anterior[x, y] = "__" # bonusul se primeste doar odata
                    elif anterior[x, y] == "+2":
                        scor_l += 2
                        anterior[x, y] = "__"

            scor += scor_l

    return scor, piese_detectate



templates_dict = {
    "cerc.png": "1",
    "trifoi.png": "2",
    "romb.png": "3",
    "patrat.png": "4",
    "stea4.png": "5",
    "stea8.png": "6",

    "bonus_1.png": "+1",
    "bonus_2.png": "+2",
}

# citim templateurile grayscale si le normalizam
template_images = {}
for fname in templates_dict.keys():
    img = cv.imread(f'templates_gray/{fname}', cv.IMREAD_GRAYSCALE)
    img_norm = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
    template_images[fname] = img_norm


# procesarea jocurilor
for joc in range(1, 6):
    # configuratia initiala a tablei
    img_start = cv.imread(f"../testare/{joc}_00.jpg")  # de schimbat cu imaginile de test
    careu = extrage_careu(img_start)
    careu_gray = cv.cvtColor(careu, cv.COLOR_BGR2GRAY)
    careu_clean = clean_padding(careu_gray, padding=25, thresh=85)

    matrice_start = determina_configuratie_careu(careu_clean, lines_horizontal, lines_vertical,
                                                 templates_dict, template_images)

    # adaugam bonusuri pe cele 4 cadrane
    matrice_start = add_bonus(matrice_start, 1, 1)
    matrice_start = add_bonus(matrice_start, 1, 9)
    matrice_start = add_bonus(matrice_start, 9, 9)
    matrice_start = add_bonus(matrice_start, 9, 1)

    # detectam culorile pentru configuratia initiala
    matrice_start = detecteaza_culori(matrice_start, careu)

    for i in range(1, 21):
        nume_imagine = f"../testare/{joc}_{i:02d}.jpg"  # de schimbat cu imaginile de test
        print(f"Incepe prelucrarea imaginii {nume_imagine}")
        nume_txt = f"../344_Cochior_Iulia-Stefana/{joc}_{i:02d}.txt"
        img = cv.imread(nume_imagine)

        careu = extrage_careu(img)
        careu_gray = cv.cvtColor(careu, cv.COLOR_BGR2GRAY)
        careu_clean = clean_padding(careu_gray, padding=25, thresh=85)
        matrice = determina_configuratie_careu(careu_clean, lines_horizontal, lines_vertical,
                                               templates_dict, template_images)

        # adaug bonusul si la imaginea actuala
        for lin in range(16):
            for col in range(16):
                # daca in matricea anterioara este bonus si in cea  actuala nu a fost detectata o piesa
                if matrice_start[lin, col] in ["+2", "+1"] and matrice[lin, col] == "__":
                    # transferam bonusul si la matricea actuala
                    matrice[lin, col] = matrice_start[lin, col]

        matrice = detecteaza_culori(matrice, careu)

        # calculam scorul si salvam in fisier
        scor, piese_detectate = calculeaza_scor(matrice_start, matrice)
        with open(nume_txt, 'w') as f:
            for piesa in piese_detectate:
                f.write(piesa + "\n")
            f.write(f"{scor}")
        matrice_start = matrice.copy()
