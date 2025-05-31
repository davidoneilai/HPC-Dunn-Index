

## **Relatório de Evolução – 2ª Parte do Trabalho Prático**

### **Comparação de Performance: Serial x Paralelo com OpenMP x CUDA na Avaliação do Índice de Dunn**

---

### **1. Objetivo do Trabalho**

O objetivo do trabalho prático é avaliar e comparar a performance do cálculo do Índice de Dunn — uma métrica utilizada para avaliar a qualidade de agrupamentos (clustering) — em três abordagens distintas: implementação sequencial (serial), paralelização com CPU utilizando OpenMP, e paralelização com GPU utilizando CUDA. O foco é evidenciar os ganhos de desempenho proporcionados por soluções paralelas em datasets de diferentes tamanhos.

---

### **2. Avanços Realizados na 2ª Parte**

Durante esta segunda fase do trabalho, foram realizados os seguintes avanços:

* Implementação de uma versão **base serial** do cálculo do Índice de Dunn em Python.
* Desenvolvimento de uma versão **paralela utilizando OpenMP**, com foco em otimizar o tempo de execução em ambientes multicore.
* Implementação de uma versão **em CUDA (via Numba/CuPy)** para aceleração em GPU.
* Execução e teste das três versões em diferentes escalas de dados:

  * Dataset **toy**: `load_iris()` do Scikit-Learn.
  * Dataset **médio**: amostra de 100 mil instâncias de um dataset de clustering do Kaggle.
  * Dataset **grande**: NYC Taxi completo, com execução planejada para ambientes de Big Data via Spark (etapa em andamento).

---

### **3. Resultados Preliminares**

Os testes demonstraram que:

* A versão **paralela com OpenMP** apresentou ganhos significativos (\~2x) em datasets médios.
* A versão **CUDA** obteve os melhores tempos em datasets maiores, apesar da complexidade inicial de configuração e otimização.
* A versão **serial** manteve boa performance apenas nos datasets pequenos.

Esses resultados reforçam a importância do paralelismo em cenários de maior escala, onde a diferença de desempenho se torna crítica.

---

### **4. Dificuldades Encontradas**

Durante o desenvolvimento, enfrentamos alguns desafios:

* Adaptação do código para múltiplos paradigmas de paralelismo (threading vs GPU).
* Controle de memória em CUDA, especialmente no tratamento de matrizes de distância.
* Integração e validação dos resultados para garantir consistência entre as versões.

Apesar disso, conseguimos contornar os principais obstáculos e validar a eficiência de cada abordagem em cenários controlados.

---

### **5. Próximos Passos**

* Realizar a **execução em ambiente distribuído (Big Data)** com o dataset completo do NYC Taxi, utilizando Spark e/ou Dask.
* Consolidar os resultados finais e elaborar gráficos de comparação (tempo, escalabilidade, uso de recursos).
* Finalizar o slide deck com os resultados comparativos e principais conclusões.

---
