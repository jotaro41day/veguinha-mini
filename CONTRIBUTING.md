# Contribuir

Abra uma issue com o problema, o comando, a seed, a duração simulada, a configuração e as versões das dependências. Para mudanças de comportamento, inclua o resultado antes/depois e um controle pareado.

Prefira testes curtos e uma mudança causal por vez. Separe atividade neural, movimento articular, apoio e deslocamento útil. Evite declarar aprendizagem com base em uma única seed ou em uma configuração inata melhor.

Não substitua o circuito por marcha prescrita, trajetória pronta, controle direto de velocidade ou recompensa por distância injetada nos DANs. Sondas de bancada devem estar explicitamente restritas aos diagnósticos.

Não publique credenciais, checkpoints pessoais, logs de sessão ou arquivos grandes de treinamento. Os resultados locais são ignorados pelo Git; novos dados pequenos e reproduzíveis podem ser adicionados intencionalmente.

Teste rápido: `python -m unittest veguinha_mini.tests.test_foveated veguinha_mini.tests.test_gait_entrypoint`.
