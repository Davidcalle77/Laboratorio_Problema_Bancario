"""
============================================================
  SIMULACIÓN DE COLAS M/M/1 - BANCO DE COLOMBIA
  Laboratorio: Problema Bancario
  Curso: Simulación de Sistemas
  Descripción: Simulación estocástica M/M/1 para optimizar
               la configuración de cajeros del banco.

  INTERPRETACIÓN DEL MODELO:
  - La tasa global de llegada (lambda_total) de cada tipo de
    acción se distribuye equitativamente entre los cajeros
    asignados a esa acción.
  - Ejemplo: si hay 2 cajeros de retiro, cada uno recibe
    lambda_retiro_total / 2 llegadas por minuto.
  - Esto garantiza rho < 1 en configuraciones adecuadas.
============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
#  1. PARÁMETROS GLOBALES DEL SISTEMA
# ─────────────────────────────────────────────────────────────

T_SIM = 480  # minutos (8 horas)
N_REP = 10  # réplicas independientes

TIPOS = ["rapido", "normal", "lento", "muy_lento"]
LABELS = {
    "rapido": "Rápido",
    "normal": "Normal",
    "lento": "Lento",
    "muy_lento": "Muy lento",
}

# Media de servicio (min) por acción y subtipo
MS = {
    "retiro": {"rapido": 1, "normal": 2, "lento": 3, "muy_lento": 4},
    "pago": {"rapido": 3, "normal": 3, "lento": 5, "muy_lento": 7},
}

# Media entre llegadas (min) — desde el punto de vista del sistema completo
ML = {
    "retiro": {"rapido": 1, "normal": 2, "lento": 3, "muy_lento": 3},
    "pago": {"rapido": 1, "normal": 2, "lento": 3, "muy_lento": 4},
}

# Probabilidades de subtipos DENTRO de cada acción
PROB = {
    "retiro": {"rapido": 0.23, "normal": 0.40, "lento": 0.17, "muy_lento": 0.20},
    "pago": {"rapido": 0.10, "normal": 0.20, "lento": 0.30, "muy_lento": 0.40},
}

# Tasa global de llegada al sistema para cada acción (clientes/min)
LAMBDA_GLOBAL = {
    accion: sum(PROB[accion][t] / ML[accion][t] for t in TIPOS)
    for accion in ["retiro", "pago"]
}

# Tasa de servicio ponderada del cajero (clientes/min)
MU_CAJERO = {
    accion: sum(PROB[accion][t] / MS[accion][t] for t in TIPOS)
    for accion in ["retiro", "pago"]
}

# Datos del sistema
DATOS_SISTEMA = [
    "3 cajas disponibles para ambas operaciones",
    "70% de usuarios realizan retiros",
    "30% de usuarios realizan pagos/consignaciones",
    "8 horas de operación diaria",
    "Modelo de colas M/M/1 para cada cajero",
    "Atención inmediata y desplazamiento despreciable",
]

print("─" * 50)
print("  Parámetros del sistema:")
for a in ["retiro", "pago"]:
    print(
        f"  {a.capitalize():8s}: λ={LAMBDA_GLOBAL[a]:.4f} cli/min | "
        f"μ={MU_CAJERO[a]:.4f} cli/min | "
        f"ρ(1 caj)={LAMBDA_GLOBAL[a]/MU_CAJERO[a]:.4f}"
    )
print("─" * 50)


# ─────────────────────────────────────────────────────────────
#  2. FUNCIÓN PRINCIPAL DE SIMULACIÓN M/M/1
# ─────────────────────────────────────────────────────────────


def simular_cajero(accion, n_cajeros_accion, seed=None):
    """
    Simula UN cajero M/M/1 para una acción dada.

    La tasa de llegada al cajero es:
        lambda_cajero = LAMBDA_GLOBAL[accion] / n_cajeros_accion

    Retorna dict con métricas de la réplica.
    """
    rng = np.random.default_rng(seed)

    # Tasa de llegada a ESTE cajero (dividida entre cajeros del mismo tipo)
    lam = LAMBDA_GLOBAL[accion] / n_cajeros_accion
    mu = MU_CAJERO[accion]
    rho = lam / mu

    # Métricas analíticas M/M/1
    if rho < 1:
        Lq = rho**2 / (1 - rho)
        Wq = rho / (mu - lam)  # tiempo esperado en cola (min)
        L = rho / (1 - rho)
        W = 1 / (mu - lam)  # tiempo esperado en sistema (min)
    else:
        Lq = Wq = L = W = float("inf")

    # ── Simulación de eventos ──────────────────────────────
    t_actual = 0.0
    t_servidor = 0.0

    conteos = {t: 0 for t in TIPOS}
    lista_espera = []
    lista_sistema = []
    lista_servicio = []

    while t_actual < T_SIM:
        # Próxima llegada (exponencial con tasa lambda del cajero)
        t_actual += rng.exponential(1.0 / lam)
        if t_actual >= T_SIM:
            break

        # Asignar subtipo según probabilidades
        subtipo = rng.choice(TIPOS, p=[PROB[accion][t] for t in TIPOS])
        conteos[subtipo] += 1

        # Tiempo de servicio exponencial del subtipo
        t_srv = rng.exponential(MS[accion][subtipo])
        lista_servicio.append(t_srv)

        # Cola y sistema
        inicio = max(t_actual, t_servidor)
        espera = inicio - t_actual
        lista_espera.append(espera)
        lista_sistema.append(espera + t_srv)
        t_servidor = inicio + t_srv

    n_total = sum(conteos.values())
    return {
        "accion": accion,
        "n_cajeros": n_cajeros_accion,
        "lam": lam,
        "mu": mu,
        "rho": rho,
        "estable": rho < 1,
        "Lq": Lq,
        "Wq_analitico": Wq,
        "L": L,
        "W_analitico": W,
        "conteos": conteos,
        "total": n_total,
        "Wq_sim": float(np.mean(lista_espera)) if lista_espera else 0.0,
        "W_sim": float(np.mean(lista_sistema)) if lista_sistema else 0.0,
        "Tsrv_sim": float(np.mean(lista_servicio)) if lista_servicio else 0.0,
    }


# ─────────────────────────────────────────────────────────────
#  3. ESCENARIOS
# ─────────────────────────────────────────────────────────────
# Cada escenario es una lista de tuplas (accion, n_cajeros_de_esa_accion)

ESCENARIOS = {
    "Esc1: Mixto (2R+1P)": [("retiro", 2), ("retiro", 2), ("pago", 1)],
    "Esc2: 1R + 2P": [("retiro", 1), ("pago", 2), ("pago", 2)],
    "Esc3: 2R + 1P": [("retiro", 2), ("retiro", 2), ("pago", 1)],
    "Esc4: 4 cajeros (2R+2P)": [("retiro", 2), ("retiro", 2), ("pago", 2), ("pago", 2)],
}


def ejecutar_escenario(config_cajeros, semilla_base=100):
    """Ejecuta N_REP réplicas de un escenario."""
    todas = []
    for r in range(N_REP):
        rep = []
        for i, (accion, n_caj) in enumerate(config_cajeros):
            seed = semilla_base + r * 100 + i * 7
            res = simular_cajero(accion, n_caj, seed=seed)
            res["cajero"] = i + 1
            res["replica"] = r + 1
            rep.append(res)
        todas.append(rep)
    return todas


# ─────────────────────────────────────────────────────────────
#  4. ANÁLISIS ESTADÍSTICO
# ─────────────────────────────────────────────────────────────


def ic95(data):
    """Media e IC 95% (t-Student)."""
    n = len(data)
    m = np.mean(data)
    se = stats.sem(data)
    ci = stats.t.interval(0.95, df=n - 1, loc=m, scale=se)
    return m, ci[0], ci[1]


def resumir(replicas):
    """Construye DataFrame con estadísticas por cajero."""
    num_cajeros = len(replicas[0])
    filas = []
    for c in range(num_cajeros):
        datos = [r[c] for r in replicas]
        wq_vals = [d["Wq_sim"] for d in datos]
        w_vals = [d["W_sim"] for d in datos]
        tot_v = [d["total"] for d in datos]
        m_wq, wq_l, wq_h = ic95(wq_vals)
        m_w, w_l, w_h = ic95(w_vals)
        filas.append(
            {
                "Cajero": c + 1,
                "Acción": datos[0]["accion"].capitalize(),
                "N cajeros tipo": datos[0]["n_cajeros"],
                "ρ": datos[0]["rho"],
                "λ caj (c/min)": datos[0]["lam"],
                "μ caj (c/min)": datos[0]["mu"],
                "Estable": datos[0]["estable"],
                "Wq_sim": m_wq,
                "Wq_IC_inf": wq_l,
                "Wq_IC_sup": wq_h,
                "W_sim": m_w,
                "W_IC_inf": w_l,
                "W_IC_sup": w_h,
                "Wq_analítico": datos[0]["Wq_analitico"],
                "W_analítico": datos[0]["W_analitico"],
                "Total_prom": np.mean(tot_v),
                "Total_std": np.std(tot_v),
                "Lq": datos[0]["Lq"],
            }
        )
    return pd.DataFrame(filas)


# ─────────────────────────────────────────────────────────────
#  5. PUNTOS SOLICITADOS
# ─────────────────────────────────────────────────────────────


def punto2(replicas_esc1):
    """Promedio de usuarios por subtipo en la totalidad de cajeros."""
    acum = {t: [] for t in TIPOS}
    acum_p = {t: [] for t in TIPOS}
    for rep in replicas_esc1:
        tot_r = {t: 0 for t in TIPOS}
        tot_p = {t: 0 for t in TIPOS}
        for caj in rep:
            if caj["accion"] == "retiro":
                for t in TIPOS:
                    tot_r[t] += caj["conteos"][t]
            else:
                for t in TIPOS:
                    tot_p[t] += caj["conteos"][t]
        for t in TIPOS:
            acum[t].append(tot_r[t])
            acum_p[t].append(tot_p[t])

    print("\n" + "=" * 58)
    print("  PUNTO 2 — Promedio de usuarios por subtipo")
    print("=" * 58)
    print(f"  {'Subtipo':12s} {'Retiro (prom ± std)':>22s} {'Pago (prom ± std)':>20s}")
    print("  " + "-" * 55)
    for t in TIPOS:
        mr = np.mean(acum[t])
        sr = np.std(acum[t])
        mp = np.mean(acum_p[t])
        sp = np.std(acum_p[t])
        print(
            f"  {LABELS[t]:12s}    {mr:6.1f} ± {sr:5.1f}          {mp:6.1f} ± {sp:5.1f}"
        )
    return acum, acum_p


def punto3(replicas_esc1):
    """Total por subtipo en cada réplica; réplica con menor total."""
    registros = []
    for rep in replicas_esc1:
        fila = {"Réplica": rep[0]["replica"]}
        total = 0
        tot_r = {t: 0 for t in TIPOS}
        tot_p = {t: 0 for t in TIPOS}
        for caj in rep:
            for t in TIPOS:
                if caj["accion"] == "retiro":
                    tot_r[t] += caj["conteos"][t]
                else:
                    tot_p[t] += caj["conteos"][t]
                total += caj["conteos"][t]
        for t in TIPOS:
            fila[f"R_{t}"] = tot_r[t]
            fila[f"P_{t}"] = tot_p[t]
        fila["Total"] = total
        registros.append(fila)

    df = pd.DataFrame(registros)
    print("\n" + "=" * 58)
    print("  PUNTO 3 — Total de usuarios por tipo en cada réplica")
    print("=" * 58)
    print(df.to_string(index=False))
    menor = df.loc[df["Total"].idxmin()]
    print(
        f"\n  → Réplica con MENOR total: Réplica {menor['Réplica']}"
        f" ({int(menor['Total'])} usuarios)"
    )
    print(
        f"    Desglose: Retiros="
        + str({t: int(menor[f"R_{t}"]) for t in TIPOS})
        + f"\n              Pagos="
        + str({t: int(menor[f"P_{t}"]) for t in TIPOS})
    )
    return df


def punto4(dfs, umbral=5.0):
    """Evalúa necesidad de cajero adicional por tiempo de espera."""
    print("\n" + "=" * 58)
    print("  PUNTO 4 — ¿Es necesario un nuevo cajero?")
    print("=" * 58)
    print(f"  Umbral Wq tolerable: {umbral} min")
    print(
        f"  {'Escenario':25s} {'Cajero':>7s} {'ρ':>6s} {'Wq sim':>8s} {'Estado':>10s}"
    )
    print("  " + "-" * 60)

    necesita = False
    for nombre, df in dfs.items():
        for _, row in df.iterrows():
            est = "EXCEDE" if row["Wq_sim"] > umbral else "OK"
            if row["Wq_sim"] > umbral:
                necesita = True
            print(
                f"  {nombre:25s} {int(row['Cajero']):>7d} "
                f"{row['ρ']:>6.3f} {row['Wq_sim']:>8.2f} {est:>10s}"
            )

    print(
        f"\n  CONCLUSIÓN: {'Se recomienda cajero adicional' if necesita else 'No es necesario aún '}"
    )
    if necesita:
        print("  Ver Esc.4 (4 cajeros) para validar la mejora.")
    return necesita


def punto5(dfs):
    """Determina la configuración óptima."""
    print("\n" + "=" * 58)
    print("  PUNTO 5 — Configuración óptima")
    print("=" * 58)
    comparacion = {}
    for nombre, df in dfs.items():
        wq_global = df["Wq_sim"].mean()
        rho_max = df["ρ"].max()
        inestable = (~df["Estable"]).any()
        comparacion[nombre] = {
            "Wq global": wq_global,
            "ρ máx": rho_max,
            "Inestable": inestable,
        }

    print(f"  {'Escenario':28s} {'Wq global':>12s} {'ρ máx':>8s} {'Sistema':>10s}")
    print("  " + "-" * 62)
    for nom, vals in comparacion.items():
        est = "INESTABLE" if vals["Inestable"] else "ESTABLE"
        print(
            f"  {nom:28s} {vals['Wq global']:>12.3f} {vals['ρ máx']:>8.3f} {est:>10s}"
        )

    # Filtrar solo estables
    estables = {k: v for k, v in comparacion.items() if not v["Inestable"]}
    if estables:
        optimo = min(estables, key=lambda k: estables[k]["Wq global"])
        print(f"\n  → CONFIGURACIÓN ÓPTIMA: {optimo}")
        print(f"    Wq global: {estables[optimo]['Wq global']:.3f} min")
        print(f"    ρ máximo:  {estables[optimo]['ρ máx']:.3f}")
    else:
        print("\n  → Todos los escenarios inestables. Agregar cajeros.")
    return optimo if estables else None


# ─────────────────────────────────────────────────────────────
#  6. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────

PALETA = {
    "retiro": "#1B5E9B",
    "pago": "#2E8B57",
    "alerta": "#E85D24",
    "neutro": "#6B7280",
    "fondo": "#F8F9FA",
}
SUBPAL = ["#1B5E9B", "#2E8B57", "#F59E0B", "#D97706"]


def graf1_wq_escenarios(dfs):
    nombres = list(dfs.keys())
    n_esc = len(nombres)
    fig, axes = plt.subplots(1, n_esc, figsize=(3.5 * n_esc, 4), sharey=False)
    if n_esc == 1:
        axes = [axes]

    fig.suptitle(
        "Tiempo de Espera en Cola (Wq) por Cajero y Escenario\n"
        "Barras: promedio simulado | Líneas: IC 95%",
        fontsize=13,
        fontweight="bold",
        y=0.95,
    )

    for ax, (nombre, df) in zip(axes, dfs.items()):
        colores = [PALETA[a.lower()] for a in df["Acción"]]
        x = np.arange(len(df))
        bars = ax.bar(
            x,
            df["Wq_sim"],
            color=colores,
            alpha=0.88,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            width=0.6,
        )
        ye_l = df["Wq_sim"] - df["Wq_IC_inf"]
        ye_h = df["Wq_IC_sup"] - df["Wq_sim"]
        ax.errorbar(
            x,
            df["Wq_sim"],
            yerr=[ye_l, ye_h],
            fmt="none",
            color="#333",
            capsize=5,
            lw=1.3,
            zorder=5,
        )
        for b, v in zip(bars, df["Wq_sim"]):
            ax.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + 0.03,
                f"{v:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [f"C{int(c)}\n({a})" for c, a in zip(df["Cajero"], df["Acción"])],
            fontsize=9,
        )
        ax.set_title(nombre.replace(" ", "\n", 1), fontsize=9, pad=6)
        ax.set_ylabel("Wq (min)", fontsize=10)
        ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        leyenda = [
            mpatches.Patch(color=PALETA["retiro"], label="Retiro"),
            mpatches.Patch(color=PALETA["pago"], label="Pago"),
        ]
        ax.legend(handles=leyenda, fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.show()
    print("grafica1_wq_cajeros.png")


def graf2_rho_escenarios(dfs):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    nombres = list(dfs.keys())
    n_esc = len(nombres)
    x_base = np.arange(n_esc)
    ancho = 0.18
    max_caj = max(len(df) for df in dfs.values())

    for c_idx in range(max_caj):
        rho_vals = []
        for df in dfs.values():
            if c_idx < len(df):
                rho_vals.append(df.iloc[c_idx]["ρ"])
            else:
                rho_vals.append(np.nan)
        offset = (c_idx - max_caj / 2 + 0.5) * ancho
        color = PALETA["retiro"] if c_idx % 2 == 0 else PALETA["pago"]
        bars = ax.bar(
            x_base + offset,
            rho_vals,
            width=ancho,
            alpha=0.85,
            color=color,
            edgecolor="white",
            label=f"Cajero {c_idx+1}",
            zorder=3,
        )
        for b, v in zip(bars, rho_vals):
            if not np.isnan(v):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    b.get_height() + 0.01,
                    f"{v:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                )

    ax.axhline(
        1.0,
        color=PALETA["alerta"],
        ls="--",
        lw=1.8,
        label="Límite estabilidad ρ=1",
        zorder=4,
    )
    ax.set_xticks(x_base)
    ax.set_xticklabels(nombres, fontsize=9, rotation=10, ha="right")
    ax.set_ylabel("Factor de utilización ρ", fontsize=11)
    ax.set_title(
        "Factor de Utilización (ρ) por Escenario y Cajero\n"
        "ρ < 1 = sistema estable | ρ ≥ 1 = sistema inestable",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper right")
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
    print("grafica2_rho.png")


def graf3_distribucion_subtipos(replicas_esc3, config):
    """Boxplots por subtipo en Escenario 3 (el más balanceado)."""
    fig, axes = plt.subplots(1, len(config), figsize=(14, 5))
    fig.suptitle(
        "Distribución de Usuarios por Subtipo — Escenario 3 (2R+1P)\n"
        "Caja bigotes sobre 10 réplicas independientes",
        fontsize=12,
        fontweight="bold",
    )

    for idx, (ax, (accion, n_caj)) in enumerate(zip(axes, config)):
        datos = {t: [rep[idx]["conteos"][t] for rep in replicas_esc3] for t in TIPOS}
        vals = [datos[t] for t in TIPOS]
        noms = [LABELS[t] for t in TIPOS]

        bp = ax.boxplot(vals, patch_artist=True, medianprops=dict(color="white", lw=2))
        for patch, col in zip(bp["boxes"], SUBPAL):
            patch.set_facecolor(col)
            patch.set_alpha(0.85)

        # Superponer puntos individuales
        for i, v in enumerate(vals, 1):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, size=len(v))
            ax.scatter(
                [i + j for j in jitter], v, color="black", alpha=0.5, s=18, zorder=5
            )

        ax.set_xticklabels(noms, rotation=15, ha="right", fontsize=9)
        ax.set_title(
            f"Cajero {idx+1} — {accion.capitalize()} ({n_caj} caj.)", fontsize=10, pad=6
        )
        ax.set_ylabel("N° usuarios", fontsize=10)
        ax.yaxis.grid(True, ls="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    print("grafica3_subtipos.png")


def graf4_totales_replicas(todas_replicas_por_esc):
    fig, ax = plt.subplots(figsize=(12, 5))
    colores_esc = [PALETA["retiro"], PALETA["pago"], PALETA["alerta"], PALETA["neutro"]]

    for (nombre, replicas), col in zip(todas_replicas_por_esc.items(), colores_esc):
        totales = [sum(c["total"] for c in rep) for rep in replicas]
        reps = list(range(1, len(totales) + 1))
        ax.plot(
            reps,
            totales,
            marker="o",
            lw=2,
            color=col,
            label=nombre,
            markersize=7,
            zorder=3,
        )
        ax.axhline(np.mean(totales), color=col, ls="--", lw=0.8, alpha=0.5)

    ax.set_xticks(range(1, N_REP + 1))
    ax.set_title(
        "Total de Usuarios Atendidos por Réplica — Todos los Escenarios",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Réplica", fontsize=11)
    ax.set_ylabel("Total usuarios atendidos", fontsize=11)
    ax.yaxis.grid(True, ls="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="lower right")
    plt.tight_layout()
    plt.show()
    print("grafica4_totales.png")


def graf5_analitico_vs_sim(df_esc3):
    """Valida simulación vs. fórmulas M/M/1 para el Esc.3 estable."""
    df_est = df_esc3[df_esc3["Estable"]]
    if df_est.empty:
        print("  [!] No hay cajeros estables para validar")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(
        "Validación: M/M/1 Analítico vs. Simulado\n(Escenario 3 — cajeros estables)",
        fontsize=12,
        fontweight="bold",
    )

    x = np.arange(len(df_est))
    w = 0.35
    labels = [f"C{int(c)}" for c in df_est["Cajero"]]

    # Wq
    ax1.bar(
        x - w / 2,
        df_est["Wq_analítico"],
        w,
        label="Analítico",
        color=PALETA["retiro"],
        alpha=0.85,
        edgecolor="white",
    )
    ax1.bar(
        x + w / 2,
        df_est["Wq_sim"],
        w,
        label="Simulado (10 rép.)",
        color=PALETA["pago"],
        alpha=0.85,
        edgecolor="white",
    )
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_title("Tiempo en cola (Wq)")
    ax1.set_ylabel("min")
    ax1.legend(fontsize=9)
    ax1.yaxis.grid(True, ls="--", alpha=0.4)
    ax1.set_axisbelow(True)

    # W
    ax2.bar(
        x - w / 2,
        df_est["W_analítico"],
        w,
        label="Analítico",
        color=PALETA["retiro"],
        alpha=0.85,
        edgecolor="white",
    )
    ax2.bar(
        x + w / 2,
        df_est["W_sim"],
        w,
        label="Simulado (10 rép.)",
        color=PALETA["pago"],
        alpha=0.85,
        edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_title("Tiempo en sistema (W)")
    ax2.set_ylabel("min")
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, ls="--", alpha=0.4)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    plt.show()
    print("grafica5_validacion.png")


def graf_info_sistema():
    """Figura separada con Datos del Sistema y Probabilidades."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.axis("off")

    title = "Datos del Sistema y Probabilidades de Tipos de Usuario"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)

    datos_txt = "Datos del Sistema:\n" + "\n".join([f"• {d}" for d in DATOS_SISTEMA])
    prob_lines = ["\nProbabilidades de Tipos (por acción):"]
    for accion in ["retiro", "pago"]:
        vals = ", ".join([f"{LABELS[t]}: {PROB[accion][t]*100:.0f}%" for t in TIPOS])
        prob_lines.append(f" {accion.capitalize()}: {vals}")

    texto_total = datos_txt + "\n" + "\n".join(prob_lines)

    fig.text(
        0.02,
        0.05,
        texto_total,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.95, edgecolor="none"),
    )

    plt.tight_layout()
    plt.show()
    print("grafica_info_sistema.png")


# ─────────────────────────────────────────────────────────────
#  7. PROGRAMA PRINCIPAL
# ─────────────────────────────────────────────────────────────


def main():
    print("\n" + "█" * 58)
    print("█  SIMULACIÓN M/M/1 — BANCO DE COLOMBIA")
    print("█  10 réplicas × 4 escenarios × 480 minutos")
    print("█" * 58)

    # ── Ejecutar réplicas ─────────────────────────────────────
    configs = {
        "Esc1: Mixto (2R+1P)": [("retiro", 2), ("retiro", 2), ("pago", 1)],
        "Esc2: 1R + 2P": [("retiro", 1), ("pago", 2), ("pago", 2)],
        "Esc3: 2R + 1P": [("retiro", 2), ("retiro", 2), ("pago", 1)],
        "Esc4: 4 cajeros (2R+2P)": [
            ("retiro", 2),
            ("retiro", 2),
            ("pago", 2),
            ("pago", 2),
        ],
    }

    print("\nEjecutando simulaciones...")
    todas = {}
    dfs = {}
    for i, (nombre, config) in enumerate(configs.items()):
        replicas = ejecutar_escenario(config, semilla_base=(i + 1) * 37)
        todas[nombre] = replicas
        dfs[nombre] = resumir(replicas)
        print(f"    {nombre}: {N_REP} réplicas OK")

    # ── Imprimir resultados por escenario ─────────────────────
    for nombre, df in dfs.items():
        print(f"\n{'─'*58}")
        print(f"  {nombre}")
        print(f"{'─'*58}")
        cols = [
            "Cajero",
            "Acción",
            "ρ",
            "λ caj (c/min)",
            "μ caj (c/min)",
            "Wq_sim",
            "W_sim",
            "Total_prom",
            "Estable",
        ]
        print(df[cols].to_string(index=False, float_format="{:.3f}".format))
        imin = df["Wq_sim"].idxmin()
        imax = df["Wq_sim"].idxmax()
        print(
            f"  → Menor espera: Cajero {int(df.loc[imin,'Cajero'])} Wq={df.loc[imin,'Wq_sim']:.3f} min"
        )
        print(
            f"  → Mayor espera: Cajero {int(df.loc[imax,'Cajero'])} Wq={df.loc[imax,'Wq_sim']:.3f} min"
        )

    # ── Puntos ────────────────────────────────────────────────
    punto2(todas["Esc3: 2R + 1P"])
    punto3(todas["Esc3: 2R + 1P"])
    punto4(dfs, umbral=5.0)
    punto5(dfs)

    # ── Gráficas ──────────────────────────────────────────────
    print("\nGenerando gráficas...")
    graf_info_sistema()
    graf1_wq_escenarios(dfs)
    graf2_rho_escenarios(dfs)
    graf3_distribucion_subtipos(todas["Esc3: 2R + 1P"], configs["Esc3: 2R + 1P"])
    graf4_totales_replicas(todas)
    graf5_analitico_vs_sim(dfs["Esc3: 2R + 1P"])

    # ── Copiar a outputs ─────────────────────────────────────

    print("\n" + "=" * 58)
    print("  SIMULACIÓN COMPLETADA EXITOSAMENTE")
    print("  Archivos en /mnt/user-data/outputs/")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    main()
