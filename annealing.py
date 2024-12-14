import numpy as np
from multiprocessing import Pool

# Verlustfunktion (Mean Squared Error, MSE) – Diese Funktion müssen Sie anpassen
# Basierend auf Ihrer Modellgleichung und den Daten aus Hathaway (2015)


# Simulated Annealing Algorithmus – Hier Ihre vollständige Implementierung einfügen
# Siehe den Algorithmus in der Projektbeschreibung und passen Sie ihn an die Anforderungen an



# Parameterkombinationen erstellen
# Basierend auf dem Projekt sollten T0 und sigma-Werte auf realistische Intervalle angepasst werden
# Zum Beispiel: T0-Werte im Bereich [100, 1000], sigma-Werte [0.8, 1.2]
T0_values = np.linspace(100, 1000, 10)  # 10 Werte zwischen 100 und 1000
sigma_values = np.linspace(0.8, 1.2, 5)  # 5 Werte zwischen 0.8 und 1.2
parameter_combinations = [(T0, sigma) for T0 in T0_values for sigma in sigma_values]

# Parallelisierungsfunktion
def parallel_task(params):
    T0, sigma = params
    # Simulated Annealing Funktion mit den aktuellen Parametern aufrufen
    loss = simulated_annealing(T0, sigma)
    return (T0, sigma, loss)

# Hauptprogramm
if __name__ == "__main__":
    # Pool mit 32 Kernen erstellen
    with Pool(32) as pool:
        # Parameterkombinationen parallel berechnen
        results = pool.map(parallel_task, parameter_combinations)
    
    # Ergebnisse sortieren
    results_sorted = sorted(results, key=lambda x: x[2])  # Sortieren nach Verlust (MSE)
    
    # Beste Parameter ausgeben
    best_params = results_sorted[0]
    print(f"Beste Parameter: T0={best_params[0]}, Sigma={best_params[1]}, Verlust={best_params[2]}")
