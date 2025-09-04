# Ui Yolo Dataset

# Progetto Metodi del Calcolo Scientifico — progetto 1 alternativo

Questo progetto analizza il comportamento dei solver iterativi (Jacobi, Gauss-Seidel, Gradiente e Gradiente coniugato) applicati su differenti matrici contenute nella directory ```/matrix/```.

## Documentazione

Una descrizione dettagliata del progetto è visualizzabile all'interno della documentazione nella directory ```/relazione/```.


## Autori

- **Federica Ratti** — 886158
- **Nicolò Molteni** — 933190

Corso di Laurea Magistrale in Informatica  
Università degli Studi di Milano-Bicocca  
A.A. 2024-2025

---

## Setup del progetto

### 1. Clona la repository

Dopo aver creato una directory spostarsi al suo interno per clonare la repository.

```bash
cd existing_repo
git clone https://github.com/niki-molte/progetto-metodi_del_calcolo_scientifico
```

### 2. Setup del virtual environmnet

Aprire la directory del progetto nel terminale ed eseguire:

```bash
    python3 -m venv venv
    source venv/bin/activate       # Linux/macOS
    venv\Scripts\activate.bat      # Windows
```

### 3. Installa le dependencies

Dopo aver attivato il virtual environment è possibile installare le dependencies.

```bash
    pip install --upgrade pip
    pip install -r requirements.txt
```  

# Run

Aprire un terminale nella stessa schermata della directory in cui è stata clonata la repository ed eseguire

```bash
    python3 main.py -path /indirizzo/file/.mtx
```  

In questo modo verranno eseguiti tutti e 4 i solver iterativi implementati su un file ```.mtx``` scelto dell'utente. Tutte le configurazioni di esecuzione possibili sono descritte all'interno della relazione.

---
