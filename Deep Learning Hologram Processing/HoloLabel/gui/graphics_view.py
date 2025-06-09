from PyQt5.QtWidgets import (
    QGraphicsView, QAction, QGraphicsRectItem, 
    QGraphicsItemGroup, QGraphicsTextItem, 
    QMenu, QGraphicsLineItem, QGraphicsEllipseItem,
)
from PyQt5.QtGui import QPen, QColor, QFont
from PyQt5.QtCore import Qt, QRectF, QPointF, QLineF, QPoint

class GraphicsView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing = False
        self.drawing_measurement = False
        self.rect_item = None
        self.measurement_item = None
        self.measurement_text_item = None
        self.measurement_items = []
        self.start_point = QPointF()
        self.end_point = QPointF()
        self.rect_item = None

        self.panning = False
        self.pan_start = QPointF()
        self.last_mouse_pos = None

        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.parent = parent  # Reference to the main window

        # Enable mouse tracking
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)

        # Mouse coordinate text item
        self.mouse_coordinate_item = QGraphicsTextItem()
        self.mouse_coordinate_item.setDefaultTextColor(QColor(0, 0, 0))  # black color
        font = QFont()
        font.setPointSize(10)
        self.mouse_coordinate_item.setFont(font)
        self.mouse_coordinate_item.setZValue(1000)  # Ensure it's on top

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ClosedHandCursor)
            self.panning = True
            self.pan_start = event.pos()
        elif self.parent.edit_mode and event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if not item or not isinstance(item, QGraphicsItemGroup):
                # Start drawing a new label
                self.start_point = self.mapToScene(event.pos())
                self.drawing = True
        elif self.parent.measurement_mode and event.button() == Qt.LeftButton:
            self.start_point = self.mapToScene(event.pos())
            self.drawing_measurement = True
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.panning:
            delta = self.pan_start - event.pos()
            self.pan_start = event.pos()
            h_scroll = self.horizontalScrollBar().value()
            v_scroll = self.verticalScrollBar().value()
            self.horizontalScrollBar().setValue(h_scroll + delta.x())
            self.verticalScrollBar().setValue(v_scroll + delta.y())
        elif self.drawing:
            self.end_point = self.mapToScene(event.pos())
            if self.rect_item:
                self.scene().removeItem(self.rect_item)
            rect = QRectF(self.start_point, self.end_point).normalized()
            pen = QPen(QColor(255, 0, 0), 2)
            self.rect_item = self.scene().addRect(rect, pen)
        elif self.drawing_measurement:
            self.end_point = self.mapToScene(event.pos())
            if not self.measurement_item:
                shape = self.parent.measurement_shape
                pen = QPen(QColor(0, 255, 0), 1)
                if shape == 'Line':
                    self.measurement_item = QGraphicsLineItem(QLineF(self.start_point, self.end_point))
                elif shape == 'Rectangle':
                    rect = QRectF(self.start_point, self.end_point).normalized()
                    self.measurement_item = QGraphicsRectItem(rect)
                elif shape == 'Ellipse':
                    rect = QRectF(self.start_point, self.end_point).normalized()
                    self.measurement_item = QGraphicsEllipseItem(rect)
                self.measurement_item.setPen(pen)
                self.scene().addItem(self.measurement_item)
            else:
                if isinstance(self.measurement_item, QGraphicsLineItem):
                    self.measurement_item.setLine(QLineF(self.start_point, self.end_point))
                elif isinstance(self.measurement_item, (QGraphicsRectItem, QGraphicsEllipseItem)):
                    rect = QRectF(self.start_point, self.end_point).normalized()
                    self.measurement_item.setRect(rect)
        else:
            super().mouseMoveEvent(event)

        # Update the mouse position
        self.update_mouse_position(event)

        # Store the last mouse position
        self.last_mouse_pos = event.pos()


    def mouseReleaseEvent(self, event):
        if self.panning and event.button() == Qt.MiddleButton:
            self.setCursor(Qt.ArrowCursor)
            self.panning = False
        elif self.drawing and event.button() == Qt.LeftButton:
            self.end_point = self.mapToScene(event.pos())
            rect = QRectF(self.start_point, self.end_point).normalized()
            current_index = self.parent.current_image_index
            if self.parent.labels:
                self.parent.labels[current_index].append({
                    'class_name': self.parent.current_class,
                    'bbox': rect
                    # No need to store graphics_item here; it will be added in redraw_labels
                })
                self.parent.statusBar().showMessage(f"Added label to image {current_index + 1}: {self.parent.current_class}")
            if self.rect_item:
                self.scene().removeItem(self.rect_item)
                self.rect_item = None
            self.drawing = False
            self.parent.redraw_labels()
        elif self.drawing_measurement and event.button() == Qt.LeftButton:
            self.end_point = self.mapToScene(event.pos())
            # Calculate dimensions
            dimensions = self.calculate_measurement()
            # Display dimensions
            self.measurement_text_item = QGraphicsTextItem(dimensions)
            self.measurement_text_item.setDefaultTextColor(QColor(0, 255, 0))
            font = QFont()
            font.setPointSize(10)
            self.measurement_text_item.setFont(font)
            self.measurement_text_item.setPos(self.end_point)
            self.scene().addItem(self.measurement_text_item)
            # Store the items for potential clearing
            self.measurement_items.append((self.measurement_item, self.measurement_text_item))
            # Reset the temporary measurement items
            self.measurement_item = None
            self.measurement_text_item = None
            self.drawing_measurement = False
        else:
            super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Zoom factor
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        # Save the scene position under the mouse cursor
        old_pos = event.pos()
        scene_pos = self.mapToScene(old_pos)
        self.centerOn(scene_pos)

        # Zoom
        if event.angleDelta().y() > 0:
            zoom_factor = zoom_in_factor
        else:
            zoom_factor = zoom_out_factor

        self.scale(zoom_factor, zoom_factor)

        delta = self.mapToScene(old_pos) - self.mapToScene(self.viewport().rect().center())
        self.centerOn(scene_pos - delta)

        # After zooming, update the position of mouse_coordinate_item
        if self.last_mouse_pos is not None:
            self.update_mouse_position(pos=self.last_mouse_pos)


    def contextMenuEvent(self, event):
        item = self.itemAt(event.pos())
        if self.parent.measurement_mode:
            menu = QMenu()
            clear_action = QAction("Clear Measurements", self)
            clear_action.triggered.connect(self.clear_measurements)

            while item and not isinstance(item, QGraphicsItemGroup):
                item = item.parentItem()
            if isinstance(item, QGraphicsItemGroup):
                size_action = QAction("Size", self)
                size_action.triggered.connect(lambda: self.parent.size_particle(item))
                
                menu.addAction(size_action)

            menu.addAction(clear_action)
            menu.exec_(event.globalPos())            
        elif self.parent.edit_mode:
            item = self.itemAt(event.pos())
            while item and not isinstance(item, QGraphicsItemGroup):
                item = item.parentItem()
            if isinstance(item, QGraphicsItemGroup):
                menu = QMenu()
                
                delete_action = QAction("Delete Label", self)
                delete_action.triggered.connect(lambda: self.parent.delete_label_item(item))
                change_action = QAction("Change Class", self)
                change_action.triggered.connect(lambda: self.parent.change_label_class(item))
            
                menu.addAction(change_action)
                menu.addAction(delete_action)

                menu.exec_(event.globalPos())
                return
        else:
            super().contextMenuEvent(event)

    def clear_measurements(self):
        # Make a copy of the list to avoid issues while modifying it during iteration
        items_to_remove = self.measurement_items.copy()
        for item, text_item in items_to_remove:
            if item is not None:
                self.scene().removeItem(item)
            if text_item is not None:
                self.scene().removeItem(text_item)
            self.measurement_items.remove((item, text_item))
        self.parent.statusBar().showMessage("All measurements cleared.")


    def calculate_measurement(self):
        shape = self.parent.measurement_shape
        try:
            pixel_size = float(self.parent.pixel_size_input.text()) / float(self.parent.magnification_input.text())
        except ValueError:
            pixel_size = 100000.0  # Default value if inputs are invalid

        if shape == 'Line':
            dx = self.end_point.x() - self.start_point.x()
            dy = self.end_point.y() - self.start_point.y()
            length_px = (dx ** 2 + dy ** 2) ** 0.5
            length_um = length_px * pixel_size
            return f"Length: {length_um:.2f} um"
        elif shape == 'Rectangle':
            width_px = abs(self.end_point.x() - self.start_point.x())
            height_px = abs(self.end_point.y() - self.start_point.y())
            width_um = width_px * pixel_size
            height_um = height_px * pixel_size
            return f"W: {width_um:.2f} um, H: {height_um:.2f} um"
        elif shape == 'Ellipse':
            # Similar to rectangle
            width_px = abs(self.end_point.x() - self.start_point.x())
            height_px = abs(self.end_point.y() - self.start_point.y())
            width_um = width_px * pixel_size
            height_um = height_px * pixel_size
            return f"Major Axis: {width_um:.2f} um, Minor Axis: {height_um:.2f} um"

    def update_mouse_position(self, event=None, pos=None):
        if self.scene() is None:
            return  # Scene not set yet

        # If the mouse_coordinate_item is not in the scene, add it
        if self.mouse_coordinate_item.scene() is None:
            self.scene().addItem(self.mouse_coordinate_item)

        if event is not None:
            pos = event.pos()
        elif pos is not None:
            pass
        else:
            return  # No position provided

        # Map the mouse position to scene coordinates
        scene_pos = self.mapToScene(pos)

        x = int(scene_pos.x())
        y = int(scene_pos.y())

        # Offset in view coordinates
        offset = QPoint(10, -10)

        # Map the mouse position plus offset to scene coordinates
        offset_pos = pos + offset
        offset_scene_pos = self.mapToScene(offset_pos)

        # Calculate the offset in scene coordinates
        offset_in_scene = offset_scene_pos - scene_pos

        # Update the text item position and content
        self.mouse_coordinate_item.setPos(scene_pos + offset_in_scene)
        self.mouse_coordinate_item.setPlainText(f"({x}, {y})")

        # Optionally, show or hide the text item based on whether the mouse is over the image
        if self.parent.current_image is not None:
            image_width = self.parent.current_image.width()
            image_height = self.parent.current_image.height()

            if 0 <= x < image_width and 0 <= y < image_height:
                self.mouse_coordinate_item.setVisible(True)
            else:
                self.mouse_coordinate_item.setVisible(False)
        else:
            self.mouse_coordinate_item.setVisible(False)

