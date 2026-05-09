% Reglas Prolog para el Tutor Inteligente de IA
% ================================================

% --- Lógica de Predicados ---

% Hechos
hombre(juan).
hombre(pedro).
mujer(maria).
mujer(ana).

progenitor(juan, pedro).
progenitor(maria, pedro).
progenitor(juan, ana).

% Reglas
padre(X, Y) :- progenitor(X, Y), hombre(X).
madre(X, Y) :- progenitor(X, Y), mujer(X).
hermano(X, Y) :- progenitor(Z, X), progenitor(Z, Y), X \= Y.

% Unificación
antecesor(X, Y) :- progenitor(X, Y).
antecesor(X, Y) :- progenitor(X, Z), antecesor(Z, Y).

% --- Búsqueda Informada (representación de grafo) ---

conectado(a, b, 1).
conectado(a, c, 4).
conectado(b, d, 2).
conectado(b, e, 5).
conectado(c, d, 1).
conectado(d, f, 3).
conectado(e, f, 2).

% Heurísticas para A*
heuristica(a, 6).
heuristica(b, 4).
heuristica(c, 4).
heuristica(d, 2).
heuristica(e, 2).
heuristica(f, 0).

% --- Aprendizaje Automático ---

% Dataset simple para clasificación
clase(perro, mamifero).
clase(gato, mamifero).
clase(ave, no_mamifero).

caracteristica(perro, [patas(4), pelos(si), vuela(no)]).
caracteristica(gato, [patas(4), pelos(si), vuela(no)]).
caracteristica(ave, [patas(2), pelos(no), vuela(si)]).

% --- Redes Neuronales (representación de arquitectura) ---

neurona(capa1_neurona1, [entrada1, entrada2], sesgo(0.5)).
neurona(capa1_neurona2, [entrada1, entrada2], sesgo(0.3)).

pesos(capa1_neurona1, entrada1, 0.7).
pesos(capa1_neurona1, entrada2, 0.2).
pesos(capa1_neurona2, entrada1, 0.4).
pesos(capa1_neurona2, entrada2, 0.6).

funcion_activacion(sigmoide).
funcion_activacion(relu).
funcion_activacion(tanh).
