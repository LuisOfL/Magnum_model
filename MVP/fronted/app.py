import flet as ft
import requests

API_URL = "http://127.0.0.1:8000/predict"


def main(page: ft.Page):
    page.title = "Detector YOLO - Flet 0.28"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.scroll = ft.ScrollMode.AUTO
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.padding = 20

    # Componentes UI
    status_text = ft.Text(
        "Selecciona una imagen de tu computadora para comenzar.",
        size=14,
        color=ft.Colors.GREY_700,
    )
    loading_ring = ft.ProgressRing(visible=False, width=30, height=30)

    img_result = ft.Image(
        width=650,
        height=450,
        fit=ft.ImageFit.CONTAIN,
        visible=False,
    )

    table_results = ft.DataTable(
        columns=[
            ft.DataColumn(label=ft.Text("Etiqueta / Producto", weight=ft.FontWeight.BOLD)),
            ft.DataColumn(label=ft.Text("Frecuencia (≥ 2)", weight=ft.FontWeight.BOLD), numeric=True),
        ],
        rows=[],
        visible=False,
    )

    def on_file_selected(e: ft.FilePickerResultEvent):
        if not e.files or len(e.files) == 0:
            return

        file_path = e.files[0].path
        
        # Mostrar estado de carga
        status_text.value = "⏳ Procesando imagen con el modelo YOLO..."
        status_text.color = ft.Colors.BLUE_700
        loading_ring.visible = True
        img_result.visible = False
        table_results.visible = False
        page.update()

        try:
            # Enviar imagen a FastAPI
            with open(file_path, "rb") as f:
                response = requests.post(
                    API_URL,
                    files={"file": (e.files[0].name, f, "image/jpeg")},
                    timeout=30,
                )

            if response.status_code == 200:
                data = response.json()

                # 1. Renderizar imagen anotada
                img_result.src_base64 = data["imagen_base64"]
                img_result.visible = True

                # 2. Renderizar tabla con frecuencias >= 2
                table_results.rows.clear()
                detecciones = data["detecciones"]

                if detecciones:
                    for item in detecciones:
                        table_results.rows.append(
                            ft.DataRow(
                                cells=[
                                    ft.DataCell(ft.Text(item["etiqueta"])),
                                    ft.DataCell(ft.Text(str(item["frecuencia"]))),
                                ]
                            )
                        )
                    table_results.visible = True
                    status_text.value = f"✅ Detección completa ({len(detecciones)} clases con frecuencia ≥ 2)."
                    status_text.color = ft.Colors.GREEN_700
                else:
                    status_text.value = "⚠️ Se detectaron objetos, pero ninguno superó la frecuencia mínima (≥ 2)."
                    status_text.color = ft.Colors.ORANGE_800

            else:
                detail = response.json().get("detail", "Error desconocido")
                status_text.value = f"❌ Error en el backend: {detail}"
                status_text.color = ft.Colors.RED_700

        except Exception as err:
            status_text.value = f"❌ No se pudo conectar con el backend: {err}"
            status_text.color = ft.Colors.RED_700

        finally:
            loading_ring.visible = False
            page.update()

    # FilePicker para Flet 0.28
    file_picker = ft.FilePicker(on_result=on_file_selected)
    page.overlay.append(file_picker)

    btn_upload = ft.ElevatedButton(
        "Subir imagen desde mi PC",
        icon=ft.Icons.UPLOAD_FILE,
        on_click=lambda _: file_picker.pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.IMAGE,
        ),
    )

    # Disposición principal
    page.add(
        ft.Column(
            controls=[
                ft.Text("Detector de Productos YOLOv8", style=ft.TextThemeStyle.HEADLINE_SMALL),
                btn_upload,
                loading_ring,
                status_text,
                ft.Divider(),
                img_result,
                table_results,
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            spacing=15,
        )
    )


if __name__ == "__main__":
    ft.app(target=main)