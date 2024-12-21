LOOP_COMPUTE_OBJ="MAXFLOW" # {"MAXUTIL","MAXFLOW","MAXCONC"}

FACTOR_MAP={
    "Brain-obj2":50,"GEANT-obj2":50,"Abilene-2-('0', '1')-('0', '2')-obj2":10,
            "Abilene-obj2-2-('0', '1')-('0', '2')":10,
            "Abilene-obj2-2-('2', '9')-('3', '4')":10,
            "Abilene-obj2-2-('3', '4')-('3', '6')":10,
            "Abilene-obj2-2-('7', '8')-('9', '10')":10,
            "Abilene-obj2-va1-2-('5','8')-('6','7')":10,
            # prog-transfer
            "Abilene-obj2-2-('5', '8')-('6', '7')":10,
            "Abilene-obj2-dis05-2-('5', '8')-('6', '7')":10,
            "Abilene-obj2-dis10-2-('5', '8')-('6', '7')":10,
            "Abilene-obj2-dis125-2-('5', '8')-('6', '7')":10,
            } # props.ecmp_topo