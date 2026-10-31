# Homeostase e flexibilidade sensorial — 2026-10-31

## Correções

A necessidade energética anterior saturava após 5% de consumo. A nova curva mantém um gradiente em toda a reserva; o déficit entra também no canal interoceptivo de valor. A fadiga passa a atenuar aferências de ativação e excitar aferências supressoras; no circuito de manutenção, fome saturada deixa de bloquear o descanso. A luz penaliza menos o repouso sob fadiga intensa. A octopamina continua modulando o ruído, agora limitado também pela fadiga.

Adicionada habituação olfatória por quatro canais bilaterais, tau8s, força1,5, com recuperação na ausência de estímulo. Necessidade e reserva modulam a saliência; novidade sensorial excita OCT. Esse filtro pertence à transdução sensorial, não é prova de memória associativa. Ele não atenua dor/contato nem escreve nos motores ou nos DANs. Nenhum cronograma comportamental, torque prescrito ou alteração da taxa-base de aprendizagem foi introduzido.

## Testes

23 testes focados passaram em35,918s, incluindo canais de valor, dor/tato, checkpoint/migração, modo de controle e competição neural. A primeira tentativa de alguns testes pelo diretório do pacote encontrou o arquivo local profile.py em vez do módulo padrão; a bateria foi repetida pela raiz correta. Não apresentar isso como falha do Brian2.

No circuito descendente existente (140células +24de permissão),3seeds42/43/44, as intervenções de fadiga alta reduziram a saída motora e a recuperação reabriu a permissão. Resultados individuais em reduced_probe.json. O circuito reduzido recebe as aferências calculadas pelo transdutor do projeto e não contém corpo. Não tratar a regularidade entre essas seeds como generalização de navegação.

## Rede e corpo completos

Cópias pareadas do mesmo checkpoint real, seed42, plasticidade motoraOFF, física2ms/rede0,5ms. Janela normal0,4s; estados cansado/recuperado1,2s cada (800ms acomodação +400ms medição). Energia95% e fadiga1/0 foram intervenções APENAS NAS CÓPIAS. A energia/fadiga da sessão viva não foram ajustadas.

| Condição | Motor antigo Hz | Motor novo Hz | Inclinação nova (cos) |
|---|---:|---:|---:|
| normal | 4.071 | 4.262 | 0.997265 |
| tired | 4.149 | 0.139 | 0.999692 |
| recovered | 3.611 | 4.661 | 0.998962 |

O corpo permaneceu estável nas janelas observadas. Isso demonstra flexibilidade fisiológica de parada/retomada, não sono consolidado, caça, navegação adaptativa ou reaprendizagem de marcha. Uma seed no corpo completo não demonstra generalização. Tempos foram medidos no25078PC3EG; durações por janela estão em full_final.json. Nenhuma simulação experimental excedeu220s reais por processo.

## Compatibilidade

Sem novos grupos neuronais ou projeções; massa, contatos, número de juntas e pesos atuais preservados. Estados dos receptores são salvos em value_flow_extra e reiniciados no novo episódio. Consultas/snapshots não avançam o filtro. O flag homeostasis_enabled e a opção --legacy-homeostasis permitem controles históricos. Checkpoints admitem a migração apenas desse flag; demais diferenças de parâmetros/topologia continuam rejeitadas.

Dados em summary.json. O protocolo reduzido publicado pode ser repetido pela raiz do repositório com `PYTHONPATH=. python veguinha_mini/results/homeostasis_v2/reduced_probe.py`. A comparação completa usa um checkpoint local de sessão, não incluído; o protocolo local chama-se full_final.py. Estado vivo em before.pkl e, após instalação, ready.pkl; checkpoints, metadados de sessão e logs ficam locais. As medições completas não representam uma execução a partir de pesos novos.
