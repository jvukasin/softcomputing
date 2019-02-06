# Soft computing - Vukašin Jović RA131/2015

Za pokretanje programa se koristio pycharm. Potrebno je pokrenuti fajl numlines.py bez dodatnih parametara prilikom pokretanja.

Proces izvršavanja programa:
- Istrenirana neuronska mreža je sačuvana u fajlu 'ann.h5', i ona se pokreće.
- Ukoliko ne postoji fajl 'ann.5' mreža se ponovo trenira i čuva u pomenutom fajlu.
- Prolazi se kroz svih 10 video klipa i vrši analiza te se rezultat štampa u 'out.txt' fajl.
- Na kraju se poziva 'test.py' fajl koji validira uspešnost rešenja nad train skupom podataka na osnovu 'res.txt' gde su zapisani tačni rezultati i fajla 'out.txt'.