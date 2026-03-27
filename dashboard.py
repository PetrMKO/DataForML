"""
Streamlit дашборд для DataForML пайплайна.

Вкладки:
  1. HITL Review  — разметка примеров из review_queue.csv, сохранение review_queue_corrected.csv
  2. Metrics      — метрики модели, learning curve, AL-отчёт, распределение классов

Запуск:
    pip install streamlit
    streamlit run dashboard.py
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="DataForML Dashboard",
    page_icon="📊",
    layout="wide",
)

REVIEW_QUEUE = Path("exports/review_queue.csv")
CORRECTED_PATH = Path("exports/review_queue_corrected.csv")
METRICS_PATH = Path("reports/final_metrics.json")
AL_REPORT_PATH = Path("reports/al_report.json")
LEARNING_CURVE = Path("reports/learning_curve.png")
ANNOTATED_DIR = Path("data/annotated")

CLASSES = ["World", "Sports", "Business", "Sci/Tech"]

# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["❗ HITL Review", "📈 Metrics"])

# ══════════════════════════════════════════════
# Вкладка 1: HITL Review
# # ══════════════════════════════════════════════
# with tab1:
#     st.title("❗ HITL-1: Проверка разметки")
#     st.caption("Проверьте и исправьте метки примеров с низкой уверенностью модели.")

#     if not REVIEW_QUEUE.exists():
#         st.warning(
#             f"Файл `{REVIEW_QUEUE}` не найден. "
#             "Запустите пайплайн до шага 3 (`AnnotationAgent`) чтобы сгенерировать очередь."
#         )
#     else:
#         df_raw = pd.read_csv(REVIEW_QUEUE)

#         # Статистика
#         col1, col2, col3 = st.columns(3)
#         col1.metric("Всего примеров", len(df_raw))
#         col2.metric("Avg confidence", f"{df_raw['confidence'].mean():.3f}" if "confidence" in df_raw.columns else "—")
#         col3.metric("Уже исправлено", len(pd.read_csv(CORRECTED_PATH)) if CORRECTED_PATH.exists() else 0)

#         st.divider()

#         # Фильтр по уверенности
#         if "confidence" in df_raw.columns:
#             min_conf, max_conf = float(df_raw["confidence"].min()), float(df_raw["confidence"].max())
#             threshold = st.slider(
#                 "Показать примеры с confidence ≤", min_conf, max_conf,
#                 value=min(0.85, max_conf), step=0.01
#             )
#             df_show = df_raw[df_raw["confidence"] <= threshold].copy()
#         else:
#             df_show = df_raw.copy()

#         st.caption(f"Показано: {len(df_show)} из {len(df_raw)} примеров")

#         # Инициализация session state для хранения правок
#         if "edited_labels" not in st.session_state:
#             st.session_state.edited_labels = {}

#         # Таблица с редактированием меток
#         st.subheader("Примеры для проверки")

#         # Detect available label classes from data
#         detected_classes = CLASSES
#         if "predicted_label" in df_raw.columns:
#             extra = [c for c in df_raw["predicted_label"].unique() if c not in CLASSES]
#             detected_classes = CLASSES + extra

#         for idx, row in df_show.iterrows():
#             with st.container():
#                 c1, c2, c3 = st.columns([5, 2, 2])

#                 text_preview = str(row.get("text", ""))[:200]
#                 c1.markdown(f"**{idx+1}.** {text_preview}{'...' if len(str(row.get('text',''))) > 200 else ''}")

#                 current_label = st.session_state.edited_labels.get(idx, row.get("predicted_label", detected_classes[0]))
#                 new_label = c2.selectbox(
#                     "Метка",
#                     options=detected_classes,
#                     index=detected_classes.index(current_label) if current_label in detected_classes else 0,
#                     key=f"label_{idx}",
#                     label_visibility="collapsed"
#                 )
#                 st.session_state.edited_labels[idx] = new_label

#                 conf = row.get("confidence", None)
#                 if conf is not None:
#                     color = "🟢" if conf >= 0.85 else "🟡" if conf >= 0.70 else "🔴"
#                     c3.markdown(f"{color} `{conf:.3f}`")

#         st.divider()

#         # Кнопка сохранения
#         col_save, col_reset = st.columns([2, 1])
#         if col_save.button("💾 Сохранить исправления → review_queue_corrected.csv", type="primary"):
#             df_result = df_show.copy()
#             df_result["predicted_label"] = [
#                 st.session_state.edited_labels.get(idx, row["predicted_label"])
#                 for idx, row in df_show.iterrows()
#             ]
#             Path("exports").mkdir(exist_ok=True)
#             df_result.to_csv(CORRECTED_PATH, index=False)
#             st.success(f"✅ Сохранено {len(df_result)} примеров → `{CORRECTED_PATH}`")
#             st.info("Теперь можно продолжить пайплайн — нажмите Enter в терминале где ждёт `human_review()`.")

#         if col_reset.button("🔄 Сбросить правки"):
#             st.session_state.edited_labels = {}
#             st.rerun()

#         # Показать уже сохранённый файл если есть
#         if CORRECTED_PATH.exists():
#             with st.expander(f"📄 Текущий `{CORRECTED_PATH}` ({len(pd.read_csv(CORRECTED_PATH))} строк)"):
#                 st.dataframe(pd.read_csv(CORRECTED_PATH), use_container_width=True)


# ══════════════════════════════════════════════
# Вкладка 2: Metrics
# ══════════════════════════════════════════════
with tab2:
    st.title("📈 Метрики пайплайна")

    # ── Итоговая модель ──────────────────────────
    st.subheader("Шаг 5 — Итоговая модель")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", metrics.get("accuracy", "—"))
        c2.metric("F1 macro", metrics.get("f1_macro", "—"))
        c3.metric("F1 weighted", metrics.get("f1_weighted", "—"))
        c4.metric("Train / Test", f"{metrics.get('n_train','—')} / {metrics.get('n_test','—')}")
    else:
        st.info("Метрики появятся после завершения Шага 5 (обучение модели).")

    st.divider()

    # ── Learning Curve ───────────────────────────
    st.subheader("Шаг 4 — Active Learning: learning curve")
    col_img, col_table = st.columns([3, 2])

    with col_img:
        if LEARNING_CURVE.exists():
            st.image(str(LEARNING_CURVE), use_container_width=True)
        else:
            st.info("График появится после завершения Шага 4 (AL-отбор).")

    with col_table:
        if AL_REPORT_PATH.exists():
            al = json.loads(AL_REPORT_PATH.read_text())
            st.metric("Лучший F1 (entropy)", f"{al.get('best_f1', '—'):.4f}" if al.get('best_f1') else "—")
            st.metric("Финальный пул", al.get("final_n_labeled", "—"))
            st.metric("Стратегия", al.get("strategy", "—"))

            history = al.get("history", [])
            if history:
                df_hist = pd.DataFrame(history)[["iteration", "n_labeled", "accuracy", "f1_macro", "strategy"]]
                st.dataframe(df_hist, use_container_width=True, hide_index=True)
        else:
            st.info("AL-отчёт появится после завершения Шага 4.")

    st.divider()

    # ── Распределение классов ────────────────────
    st.subheader("Шаг 3 — Распределение меток в аннотированном датасете")
    annotated_files = sorted(ANNOTATED_DIR.glob("*.parquet")) if ANNOTATED_DIR.exists() else []
    if annotated_files:
        df_ann = pd.read_parquet(annotated_files[-1])
        label_col = "predicted_label" if "predicted_label" in df_ann.columns else "label"
        if label_col in df_ann.columns:
            col_chart, col_stats = st.columns([2, 1])
            vc = df_ann[label_col].value_counts()
            col_chart.bar_chart(vc)
            col_stats.dataframe(
                vc.reset_index().rename(columns={label_col: "Класс", "count": "Кол-во"}),
                hide_index=True, use_container_width=True
            )

        col_conf1, col_conf2 = st.columns(2)
        if "confidence" in df_ann.columns:
            col_conf1.metric("Avg confidence", f"{df_ann['confidence'].mean():.4f}")
            col_conf2.metric(
                "Флагнуто для HITL",
                f"{(df_ann['confidence'] < 0.85).sum()} ({(df_ann['confidence'] < 0.85).mean()*100:.1f}%)"
            )
    else:
        st.info("Данные появятся после завершения Шага 3 (авторазметка).")
