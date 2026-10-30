# Ladybird: pele visual da Veguinha

- Original: [Ladybird, de Exceptional_3D](https://poly.pizza/m/3tCnJC9UYA), licença **Public Domain (CC0)** informada na página do modelo. Atribuição não é obrigatória.
- Arquivo baixado: `ladybird_original.glb` (SHA-256 em `conversion.json`).
- GLB: nós `RootNode` e `Sphere`; um mesh `Sphere` com três primitivas de material `black.001`, `black` e `red`. Não há nós de pernas ou juntas. A malha mostra casco, cabeça e antenas; as seis pernas são desenhadas por cápsulas locais do MJCF.
- Conversão: a cena GLB foi carregada com `trimesh`; cada primitiva virou um OBJ no referencial do tórax. Após aplicar a transformação da cena, as coordenadas em metros são `x=0.13*z+0.15`, `y=0.08*x`, `z=0.10*y+0.025`. Agrupamento de vértices em grade de 5 mm removeu faces degeneradas e duplicadas. Foram de 8.662 a 5.487 triângulos (−36,7%). Contagens por arquivo e limites estão em `conversion.json`.
- Uso: os três OBJs são presos ao corpo `thorax`; as cápsulas afiladas das pernas seguem os elos `*_femur` e `*_tibia`. Todos os geoms novos usam `group=1`, `mass=0`, `contype=0`, `conaffinity=0`. Os geoms físicos antigos continuam presentes com transparência visual; motores, sensores, contatos, massa e inércia não foram alterados.
