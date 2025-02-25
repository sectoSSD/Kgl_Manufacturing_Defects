

class Vehicule:

    def __init__(self, genre, poid, vitesse, autonomie, passager, contenance):
        self.genre=genre
        self.poid=poid
        self.vitesse=vitesse
        self.autonomie=autonomie
        self.passager=passager
        self.contenance=contenance

    def deplacement(self):
        print(f"le/la {self.genre} se deplace à une vitesse de {self.vitesse} km/h")

vehicule1=Vehicule('voiture', 2, 250, 1000, 5, 4)

vehicule1.deplacement()

class Voiture(Vehicule):

    def __init__(self, marque, modèle, *args, **kwargs):
        self.marque=marque
        self.modèle=modèle

vehicule1=Voiture('Volvo', 'XC70', 'voiture', 2, 250, 1000, 5, 4)
