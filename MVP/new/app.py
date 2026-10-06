import flet as ft
import requests

API_URL = "http://127.0.0.1:8000/predict"

def main(page: ft.Page):
    # 1. Configuración principal de la ventana (Modo Oscuro)
    page.title = "Smart Freezer Audit - IA"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0B0F19"  # Fondo principal súper oscuro (Azul noche/Negro)
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 20
    page.scroll = ft.ScrollMode.AUTO

    # 2. Barra de Navegación Superior (AppBar) elegante
    page.appbar = ft.AppBar(
        leading=ft.Icon(ft.Icons.ICECREAM, color="#38BDF8", size=28),
        title=ft.Text("Ejemplo de mini mini modelo", weight=ft.FontWeight.W_600, color=ft.Colors.WHITE),
        center_title=True,
        bgcolor="#111827", # Gris oscuro contrastante
        elevation=2,
    )

    # 3. Componentes visuales
    
    # 3.1 Contenedor de Alertas (Éxito / Error)
    alert_banner = ft.Container(
        content=ft.Text("", size=15, weight=ft.FontWeight.W_500),
        padding=15,
        border_radius=10,
        visible=False,
        alignment=ft.alignment.center,
        border=ft.border.all(1, ft.Colors.TRANSPARENT)
    )

    # 3.2 Indicador de Carga Animado
    loading_ring = ft.Container(
        content=ft.Row(
            controls=[
                ft.ProgressRing(width=20, height=20, stroke_width=3, color="#38BDF8"),
                ft.Text("Analizando profundidad e inventario...", size=16, color="#9CA3AF", weight=ft.FontWeight.W_400)
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            spacing=15
        ),
        visible=False,
        margin=ft.margin.symmetric(vertical=15)
    )

    # 3.3 Tarjeta para la Imagen Resultante
    img_result = ft.Image(
        width=700,
        height=500,
        fit=ft.ImageFit.CONTAIN,
        border_radius=ft.border_radius.all(10),
    )
    img_card = ft.Container(
        content=img_result,
        bgcolor="#1F2937",  # Superficie elevada
        border_radius=15,
        padding=10,
        border=ft.border.all(1, "#374151"), # Borde sutil
        shadow=ft.BoxShadow(blur_radius=20, spread_radius=2, color="#000000"),
        visible=False,
    )

    # 3.4 Tarjeta para la Tabla de Resultados
    table_results = ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text("Producto Identificado", weight=ft.FontWeight.BOLD, color="#E5E7EB")),
            ft.DataColumn(label=ft.Text("Stock Estimado (Piezas)", weight=ft.FontWeight.BOLD, color="#E5E7EB"), numeric=True),
        ],
        rows=[],
        width=600,
        heading_row_color="#374151",
        border_radius=10,
        divider_thickness=0.5,
    )
    table_card = ft.Container(
        content=ft.Column(
            controls=[
                ft.Text("Resumen de Inventario", size=20, weight=ft.FontWeight.W_600, color="#38BDF8"),
                ft.Divider(color="#374151"),
                table_results
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER
        ),
        bgcolor="#1F2937",
        border_radius=15,
        border=ft.border.all(1, "#374151"),
        padding=25,
        shadow=ft.BoxShadow(blur_radius=20, spread_radius=2, color="#000000"),
        visible=False,
        width=700
    )

    # 4. Lógica de subida y conexión con la API
    def on_file_selected(e: ft.FilePickerResultEvent):
        if not e.files or len(e.files) == 0:
            return

        file_path = e.files[0].path
        
        img_card.visible = False
        table_card.visible = False
        alert_banner.visible = False
        loading_ring.visible = True
        page.update()

        try:
            with open(file_path, "rb") as f:
                response = requests.post(
                    API_URL,
                    files={"file": (e.files[0].name, f, "image/jpeg")},
                    timeout=45,
                )

            if response.status_code == 200:
                data = response.json()
                
                img_result.src_base64 = data["imagen_base64"]
                img_card.visible = True

                table_results.rows.clear()
                detecciones = data["detecciones"]

                if detecciones:
                    total_piezas = 0
                    for item in detecciones:
                        total_piezas += item["frecuencia"]
                        table_results.rows.append(
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text(item["etiqueta"], color="#D1D5DB")),
                                    ft.DataCell(ft.Text(str(item["frecuencia"]), weight=ft.FontWeight.BOLD, color="#34D399")),
                                ]
                            )
                        )
                    table_card.visible = True
                    
                    alert_banner.content.value = f"✅ Auditoría completada: {total_piezas} piezas estimadas."
                    alert_banner.bgcolor = "#064E3B"
                    alert_banner.border = ft.border.all(1, "#059669")
                    alert_banner.content.color = "#6EE7B7"
                else:
                    alert_banner.content.value = "⚠️ Congelador detectado vacío o productos irreconocibles."
                    alert_banner.bgcolor = "#451A03"
                    alert_banner.border = ft.border.all(1, "#D97706")
                    alert_banner.content.color = "#FCD34D"

            else:
                detail = response.json().get("detail", "Error desconocido")
                alert_banner.content.value = f"❌ Error en el backend: {detail}"
                alert_banner.bgcolor = "#450A0A"
                alert_banner.border = ft.border.all(1, "#DC2626")
                alert_banner.content.color = "#FCA5A5"

        except Exception as err:
            alert_banner.content.value = f"❌ No se pudo conectar con el servidor: {err}"
            alert_banner.bgcolor = "#450A0A"
            alert_banner.border = ft.border.all(1, "#DC2626")
            alert_banner.content.color = "#FCA5A5"

        finally:
            alert_banner.visible = True
            loading_ring.visible = False
            page.update()

    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)

    upload_zone = ft.Container(
        content=ft.Column(
            controls=[
                ft.Icon(ft.Icons.CLOUD_UPLOAD_OUTLINED, size=55, color="#38BDF8"),
                ft.Text("Haz clic aquí para subir una foto", size=18, weight=ft.FontWeight.W_500, color="#E5E7EB"),
                ft.Text("Soporta JPG y PNG", size=13, color="#6B7280"),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        ),
        width=700,
        height=180,
        bgcolor="#1F2937",
        border_radius=15,
        border=ft.border.all(2, "#374151"),
        ink=True,
        on_click=lambda _: file_picker.pick_files(allow_multiple=False, file_type=ft.FilePickerFileType.IMAGE),
        shadow=ft.BoxShadow(blur_radius=15, color="#000000"),
        animate=ft.Animation(duration=300, curve=ft.AnimationCurve.EASE_OUT)  # <--- Corregido para Flet 0.28
    )

    def on_hover_upload(e):
        e.control.border = ft.border.all(2, "#38BDF8" if e.data == "true" else "#374151")
        e.control.update()
    upload_zone.on_hover = on_hover_upload

    page.add(
        ft.Column(
            controls=[
                ft.Container(height=15),
                upload_zone,
                loading_ring,
                alert_banner,
                img_card,
                table_card,
                ft.Container(height=30),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=25,
        )
    )

if __name__ == "__main__":
    ft.app(target=main)