import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import np_utils
import math


def scale_to_range(image):  # skalira elemente slike sa 0-255 na opseg 0-1
    return image / 255


def invert(image):
    return 255 - image


def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def matrix_to_vector(image):  # sliku 28x28 transformisati u vektor sa 784 elementa
    return image.flatten()


def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    ret, image_bin = cv2.threshold(image_gs, 127, 255, cv2.THRESH_BINARY)
    return image_bin


def winner(output):  # output je vektor sa izlaza neuronske mreze, pronalazi najvise pobudjen neuron
    return max(enumerate(output), key=lambda x: x[1])[0]


def display_result(outputs, alphabet):
    '''za svaki rezultat pronaći indeks pobedničkog
        regiona koji ujedno predstavlja i indeks u alfabetu.
        Dodati karakter iz alfabet u result'''
    result = []
    for output in outputs:
        result.append(alphabet[winner(output)])
    return result


def prepare_for_ann(inputs):
    '''Regioni su matrice dimenzija 28x28 čiji su elementi vrednosti 0 ili 255.
        Potrebno je skalirati elemente regiona na [0,1] i transformisati ga u vektor od 784 elementa '''
    ready_for_ann = []
    for input in inputs:
        # skalirati elemente regiona
        # region sa skaliranim elementima pretvoriti u vektor
        # vektor dodati u listu spremnih regiona
        scale = scale_to_range(input)
        ready_for_ann.append(matrix_to_vector(scale))

    return ready_for_ann


def convert_output(alphabet):
    '''Konvertovati alfabet u niz pogodan za obučavanje NM,
        odnosno niz čiji su svi elementi 0 osim elementa čiji je
        indeks jednak indeksu elementa iz alfabeta za koji formiramo niz.
        Primer prvi element iz alfabeta [1,0,0,0,0,0,0,0,0,0],
        za drugi [0,1,0,0,0,0,0,0,0,0] itd..
    '''
    nn_outputs = []
    asd = len(alphabet)
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann():  # NM sa 784 neurona na ulaznom sloju, medjuslojem i 10 neurona na izlaznom sloju
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dropout(0.2))
    ann.add(Dense(10, activation='softmax'))  # softmax se koristi u vecini slucajeva kada imamo one-hot prezentaciju brojeva
    return ann


def treniraj(ann, X_train, y_train):
    X_train = np.array(X_train, np.float32)  # dati ulazi
    y_train = np.array(y_train, np.float32)  # zeljeni izlazi za date ulaze

    # definisanje parametra algoritma za obucavanje
    sgd = SGD(lr=0.01, momentum=0.9)
    # ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])

    # obucavanje neuronske mreze
    ann.fit(X_train, y_train, batch_size=256, epochs=100, verbose=1, shuffle=False)

    return ann


def resize_region(region):  # Transformisati selektovani region na sliku dimenzija 28x28
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)


def select_roi(image_orig, image_bin):
    '''Oznaciti regione od interesa na originalnoj slici. (ROI = regions of interest)
        Za svaki region napraviti posebnu sliku dimenzija 28 x 28.
        Za označavanje regiona koristiti metodu cv2.boundingRect(contour).
        Kao povratnu vrednost vratiti originalnu sliku na kojoj su obeleženi regioni
        i niz slika koje predstavljaju regione sortirane po rastućoj vrednosti x ose
    '''
    img, contours, hierarchy = cv2.findContours(image_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_regions = []  # lista sortiranih regiona po x osi (sa leva na desno)
    regions_array = []
    koord = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)  # koordinate i velicina granicnog pravougaonika
        area = cv2.contourArea(contour)
        if (area > 15 and h < 45 and h > 13 and w > 1) or (area > 15 and h < 45 and h > 9 and w > 12):
            # kopirati [y:y+h+1, x:x+w+1] sa binarne slike i smestiti u novu sliku
            # označiti region pravougaonikom na originalnoj slici (image_orig) sa rectangle funkcijom
            # region = image_bin[y:y+h+1, x:x+w+1]
            # region = image_bin[y-3:y+h+3, x-4:x-4+w+8] # za bolji prikaz odvojenih slika
            region = image_bin[y:y + h + 1, x:x + w + 1]
            regions_array.append([resize_region(region), (x, y, w, h)])
            koord.append([x, y, w, h])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)
    regions_array = sorted(regions_array, key=lambda item: item[1][0])
    sorted_regions = sorted_regions = [region[0] for region in regions_array]
    koord = sorted(koord, key=lambda item: item[0])

    # sortirati sve regione po x osi (sa leva na desno) i smestiti u promenljivu sorted_regions
    return image_orig, sorted_regions, koord


def spremi_broj(im):
    ret, im_bin = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)
    img, contours, hierarchy = cv2.findContours(im_bin.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 15 and h < 40 and h > 13 and w > 1:
            gotova = im_bin[y:y + h + 1, x:x + w + 1]
            gotova = resize_region(gotova)
            return gotova

    return im


def spremni_ulaze(img):
    for i in range(len(img)):
        img[i] = spremi_broj(img[i])

    return img


def NeuronskaMreza():
    # the data, shuffled and split between tran and test sets
    (input_train, output_train), (input_test, output_test) = mnist.load_data()

    input_train = spremni_ulaze(input_train)
    # input_test = spremni_ulaze(input_test)

    # pripremanje inputa i outputa u odgovarajucu formu i treniranje NM
    inputs = prepare_for_ann(input_train)
    outputs = np_utils.to_categorical(output_train, 10)
    ann = create_ann()
    ann = treniraj(ann, inputs, outputs)

    ann.save('ann.h5')

    del ann


def otvaranje(frm):
    kernel = np.ones((3, 3))
    img_ero = cv2.erode(frm, kernel, iterations=1)
    frm = cv2.dilate(img_ero, kernel, iterations=1)
    return frm


def koordinate_linije(niz):
    x1, y1, x2, y2 = niz[0]
    for n in niz:
        xa, ya, xb, yb = n
        if yb < y2:
            x1 = xa
            y1 = ya
            x2 = xb
            y2 = yb
    return x1, y1, x2, y2

def detektuj_linije_i_koordinate(f):
    hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)

    # granice za zelenu i plavu boju
    lower_green = np.array([60, 100, 100])
    upper_green = np.array([60, 255, 255])
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # maska i rezultat
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    res_green = cv2.bitwise_and(f, f, mask=mask_green)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    res_blue = cv2.bitwise_and(f, f, mask=mask_blue)

    res_greenGray = cv2.cvtColor(res_green, cv2.COLOR_BGR2GRAY)
    res_blueGray = cv2.cvtColor(res_blue, cv2.COLOR_BGR2GRAY)

    # nadji linije
    edgesGreen = cv2.Canny(res_greenGray, 50, 150, apertureSize=3)
    edgesBlue = cv2.Canny(res_blueGray, 50, 150, apertureSize=3)
    linesGreen = cv2.HoughLinesP(edgesGreen, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=18)
    linesBlue = cv2.HoughLinesP(edgesBlue, 1, np.pi / 180, 50, minLineLength=10, maxLineGap=18)

    # nadji koordinate pa oboji samo jednu koja je srednja vrednost, za obe linije
    green_line = []
    blue_line = []
    for line in linesGreen:
        x1, y1, x2, y2 = line[0]
        green_line.append(line[0])
    x1, y1, x2, y2 = koordinate_linije(green_line)
    gr_line_coord =[x1, y1, x2, y2]
    # cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

    for line in linesBlue:
        x1, y1, x2, y2 = line[0]
        blue_line.append(line[0])
    x1, y1, x2, y2 = koordinate_linije(blue_line)
    bl_line_coord = [x1, y1, x2, y2]
    # cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    return gr_line_coord, bl_line_coord


def nadji_cosak(koord):
    rez = []
    for coo in koord:
        x, y, w, h = coo
        rez.append([x+w, y+h]) # donji desni cosak regiona
    return rez


def vector(b, e):
    x, y = b
    X, Y = e
    return X-x, Y-y


def duzina(v):
    x, y = v
    return math.sqrt(x * x + y * y)


def distanca(p0, p1):
    return duzina(vector(p0, p1))


def detektujKoliziju(coord,m,c,s1,s2):
    x1, y1, x2, y2 = coord
    if x2+2>=s1>=x1-5 and y1+5>=s2>=y2-1:
        d = m*s1 + c
        if abs(int(s2)-int(d))<= 2: # 1.6 je bilo umesto 2
            return True
    return False

def equasion(niz):
    x1, y1, x2, y2 = niz
    x = [x1, x2]
    y = [y1, y2]
    coefficients = np.polyfit(x, y, 1)
    return coefficients[0], coefficients[1]


def broj_postoji(n, broj, brojevi, vrednost):
    ret = []
    for b in brojevi:
        dist = distanca(broj['koord'], b['koord'])
        if dist < n and vrednost == b['vrednost']:
            ret.append(b)
    return ret



