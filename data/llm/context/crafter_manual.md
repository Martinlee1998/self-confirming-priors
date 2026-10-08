# Crafter Manual Context v1 for ALPAGD-RL LLM Prior Capture

Source status: public Crafter source/manual summary for G6 prompt rendering. This is not a reference graph and must not be treated as edge truth.

Sources checked on 2026-05-25:

- `https://github.com/danijar/crafter`
- `https://raw.githubusercontent.com/danijar/crafter/main/crafter/data.yaml`
- `https://raw.githubusercontent.com/danijar/crafter/main/README.md`

## Environment Summary

Crafter is an open-world 2D survival environment. The player explores randomly generated worlds, forages for food and water, sleeps, defends against monsters, collects materials, places objects, and crafts tools. Evaluation is based on 22 semantically meaningful achievements.

## Achievement IDs

- collect_coal
- collect_diamond
- collect_drink
- collect_iron
- collect_sapling
- collect_stone
- collect_wood
- defeat_skeleton
- defeat_zombie
- eat_cow
- eat_plant
- make_iron_pickaxe
- make_iron_sword
- make_stone_pickaxe
- make_stone_sword
- make_wood_pickaxe
- make_wood_sword
- place_furnace
- place_plant
- place_stone
- place_table
- wake_up

## Actions

The action set includes movement, a general do/use action, sleep, placing stone/table/furnace/plant, and crafting wood/stone/iron pickaxes and swords.

## Materials And Items

Relevant materials include water, grass, stone, path, sand, tree, lava, coal, iron, diamond, table, and furnace. Relevant inventory items include health, food, drink, energy, sapling, wood, stone, coal, iron, diamond, pickaxes, and swords.

## Collection Rules

- Collecting tree gives wood and does not require a tool.
- Collecting stone requires a wood pickaxe and gives stone.
- Collecting coal requires a wood pickaxe and gives coal.
- Collecting iron requires a stone pickaxe and gives iron.
- Collecting diamond requires an iron pickaxe and gives diamond.
- Collecting water gives drink and does not require a tool.
- Interacting with grass can give sapling with low probability.

## Placement Rules

- Placing stone consumes stone and can place it on grass, sand, path, water, or lava.
- Placing a table consumes wood and can place it on grass, sand, or path.
- Placing a furnace consumes stone and can place it on grass, sand, or path.
- Placing a plant consumes sapling and requires grass.

## Crafting Rules

- Making a wood pickaxe consumes wood and requires being near a table.
- Making a wood sword consumes wood and requires being near a table.
- Making a stone pickaxe consumes wood and stone and requires being near a table.
- Making a stone sword consumes wood and stone and requires being near a table.
- Making an iron pickaxe consumes wood, coal, and iron and requires being near both a table and a furnace.
- Making an iron sword consumes wood, coal, and iron and requires being near both a table and a furnace.

## Ambiguous Or Contextual Achievements

Combat, eating, sleep, and wake-up achievements depend on game context not fully specified by the compact rules above. Treat these relations as uncertain unless the manual context directly supports a stronger dependency.
