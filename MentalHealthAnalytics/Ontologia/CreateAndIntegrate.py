import os
import pandas as pd
from rdflib import Graph, Literal, RDF, URIRef, Namespace
from rdflib.namespace import XSD, OWL

# Carica il dataset
current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(current_dir, '..', 'Risultati', 'DisturbiMentali-DalysNazioniDelMondo-GruppoDiIntervento.csv')

df = pd.read_csv(file_path)

# namespace
ONTO = Namespace("http://example.org/ontology/mental-health#")
OBO = Namespace("http://purl.obolibrary.org/obo/")

# Mappatura dei disturbi ai loro URI
disorder_uris = {
    "Schizophrenia": OBO.DOID_5419,
    "Depressive": OBO.DOID_1596,
    "Anxiety": OBO.DOID_2030,
    "Bipolar": OBO.DOID_3312,
    "Eating": OBO.DOID_8670
}

# Crea un nuovo grafo per l'ontologia integrata
g = Graph()

# Associa i namespace
g.bind("onto", ONTO)
g.bind("obo", OBO)

# Importa l'ontologia esistente
existing_ontology_path = os.path.join(current_dir, 'HumanDiseaseOntology.owl')
g.parse(existing_ontology_path)


# Funzione per creare RDF
def create_rdf_triples(row):
    country = URIRef(ONTO + row['Entity'].replace(" ", "_"))
    year = Literal(row['Year'], datatype=XSD.gYear)
    group = row[
        'Gruppo_di_intervento (0: "sviluppo economico: medio livelli alti di depressione e ansia", 1: "reddito alto: prevalenza disturbi depressivi", 2: "Reddito basso: prevalenza di disturbi di ansia, bipolare e schizofrenico")']

    g.add((country, RDF.type, ONTO.Country))
    g.add((country, ONTO.hasYear, year))

    for disorder, uri in disorder_uris.items():
        g.add((country, ONTO.hasDisorder, uri))
    if group == 0:
        group_label = ONTO.EconomicDevelopment
    elif group == 1:
        group_label = ONTO.HighIncome
    else:
        group_label = ONTO.LowIncome

    g.add((country, ONTO.belongsToGroup, group_label))

    # Aggiungi altre colonne come proprietà
    g.add((country, ONTO.schizophreniaDisorders, Literal(row['Schizophrenia disorders'])))
    g.add((country, ONTO.depressiveDisorders, Literal(row['Depressive disorders'])))
    g.add((country, ONTO.anxietyDisorders, Literal(row['Anxiety disorders'])))
    g.add((country, ONTO.bipolarDisorders, Literal(row['Bipolar disorders'])))
    g.add((country, ONTO.eatingDisorders, Literal(row['Eating disorders'])))
    g.add((country, ONTO.dalysDepressiveDisorders, Literal(row['DALYs Cause: Depressive disorders'])))
    g.add((country, ONTO.dalysSchizophrenia, Literal(row['DALYs Cause: Schizophrenia'])))
    g.add((country, ONTO.dalysBipolarDisorder, Literal(row['DALYs Cause: Bipolar disorder'])))
    g.add((country, ONTO.dalysEatingDisorders, Literal(row['DALYs Cause: Eating disorders'])))
    g.add((country, ONTO.dalysAnxietyDisorders, Literal(row['DALYs Cause: Anxiety disorders'])))
    g.add((country, ONTO.clusterKMeans, Literal(row['Cluster_KMeans'])))


# Crea RDF per tutte le righe del dataset
df.apply(create_rdf_triples, axis=1)

# Serializza il grafo in un file OWL
output_path = os.path.join(current_dir, '..', 'Risultati', 'IntegratedOntology.owl')
g.serialize(destination=output_path, format='xml')
