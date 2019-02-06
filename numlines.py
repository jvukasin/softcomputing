from funkcije import *
from keras.models import load_model
import matplotlib.pyplot as plt
import os


def klasifikuj_broj(slika, alphabet, ann):
    temp = np.array(slika)
    inp = prepare_for_ann(temp)
    result = ann.predict(np.array(inp, np.float32))
    rez = display_result(result, alphabet)[0]
    return rez


def broj_postoji(n, broj, brojevi):
    ret = []
    for b in brojevi:
        dist = distanca(broj['koord'], b['koord'])
        if dist < n:
            ret.append(b)
    return ret



def Registrovan(c):
    global brojeviFrejma
    x, y = c
    broj = {'koord': (x, y)}
    postoji = broj_postoji(10, broj, brojeviFrejma)
    dp = len(postoji)
    if dp == 0:
        return False
    else:
        return True


def program(putanja, ann):

    # ucitavanje videa
    frame_num = 0
    cap = cv2.VideoCapture(putanja)
    cap.set(1, frame_num)  # indeksiranje frejmova
    ret_val, frame = cap.read()

    frame2 = otvaranje(frame)

    gr_line_coord, bl_line_coord = detektuj_linije_i_koordinate(frame2) # pronadji linije za prvi frejm, na ostalima su isti

    mG, cG = equasion(gr_line_coord)
    mB, cB = equasion(bl_line_coord)

    global brojeviFrejma
    global idC
    rezultat = 0
    idC = -1
    brojeviFrejma = []
    addArray = []
    subArray = []
    resetujBrojeve = 0
    pomocna = None
    while True:
        frame_num += 1
        ret_val, frame = cap.read()
        # ako frejm nije zahvacen
        if not ret_val:
            break

        # rad sa frejmom
        # fm = ukloni_linije(frame)
        frejm = image_bin(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        selected_regions, numbers, koord = select_roi(frame.copy(), frejm)  # frame.copy(), frejm
        coskovi = nadji_cosak(koord)


        if resetujBrojeve == 15:  # 20 je bilo ok ali je neke brojeve ponavljao
            resetujBrojeve = 0
            for br in brojeviFrejma:
                pomocna = br
            brojeviFrejma = []

        # cv2.imshow('asd', selected_regions)  # za iscrtavanje celog videa sa regionima
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        numBroj = 0
        for c in coskovi:
            s1, s2 = c
            plava = detektujKoliziju(bl_line_coord, mB, cB, s1, s2)
            if plava == True:
                if len(brojeviFrejma) == 0 and pomocna is not None:
                    brojeviFrejma.append(pomocna)
                reg = Registrovan(c)
                if reg == False:
                    broj = {'koord': (s1, s2)}
                    brojeviFrejma.append(broj)
                    vrednost = klasifikuj_broj([numbers[numBroj]], alphabet, ann)
                    # plt.imshow(numbers[numBroj], 'gray')
                    # plt.show()
                    # print(vrednost)
                    # plt.imshow(frame)
                    # plt.show()
                    rezultat += vrednost
                    addArray.append(vrednost)
            zelena = detektujKoliziju(gr_line_coord, mG, cG, s1, s2)
            if zelena == True:
                if len(brojeviFrejma) == 0 and pomocna is not None:
                    brojeviFrejma.append(pomocna)
                reg = Registrovan(c)
                if reg == False:
                    broj = {'koord': (s1, s2)}
                    brojeviFrejma.append(broj)
                    vrednost = klasifikuj_broj([numbers[numBroj]], alphabet, ann)
                    # plt.imshow(numbers[numBroj], 'gray')
                    # plt.show()
                    # print(vrednost)
                    # plt.imshow(frame)
                    # plt.show()
                    rezultat -= vrednost
                    subArray.append(vrednost)
            numBroj += 1
        resetujBrojeve += 1
    cap.release()
    # print('Added = {}, Substracted = {}'.format(str(addArray), str(subArray)))
    return rezultat


if os.path.exists('ann.h5') is False:
    NeuronskaMreza()

ann = load_model('ann.h5')

alphabet = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

results = []
naziviVid = []
for i in range(10):
    putanja = 'video-' + str(i) + '.avi'
    naziviVid.append(putanja)
    result = program(putanja=putanja, ann=ann)
    print(result)
    results.append(result)

f = open('out.txt', 'w')
tekst = 'RA 131/2015 Vukasin Jovic\nfile\tsum\n'

for i in range(10):
    tekst += naziviVid[i] + '\t' + str(results[i]) + '\n'

f.write(tekst)
f.close()

import test


