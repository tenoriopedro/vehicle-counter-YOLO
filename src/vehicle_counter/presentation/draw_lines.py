import cv2
import numpy as np
import numpy.typing as npt


class LineCalibrator:
    """
    Interactive GUI tool utilizing OpenCV to establish and capture a counting line
    for vehicle tracking. Encapsulates window state and mouse event callbacks.
    """

    def __init__(self, window_name: str = "Line Calibration Tool") -> None:

        self.line_points: list[tuple[int, int]] = []
        self.window_name = window_name
        self.frame: npt.NDArray[np.uint8] | None = None

    def handle_click(
        self, event: int, x: int, y: int, flags: int, param: object
    ) -> None:
        """
        OpenCV mouse callback event handler. Captures left-click coordinates
        and updates the visual representation of the calibration line.
        """

        if self.frame is None:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"[NEW COORDINATE] X: {x} | Y: {y}")

            if len(self.line_points) < 2:
                self.line_points.append((x, y))

            else:
                self.line_points.clear()
                self.line_points.append((x, y))

            frame_temp = self.frame.copy()

            if len(self.line_points) == 2:
                point1, point2 = self.line_points
                cv2.line(frame_temp, point1, point2, (0, 255, 0), 5)

            for point in self.line_points:
                # Draw a green dot ai the clicked location for visual feedback
                cv2.circle(frame_temp, point, 5, (0, 255, 0), -1)

            cv2.imshow(self.window_name, frame_temp)

    def run(self, frame: npt.NDArray[np.uint8]) -> list[tuple[int, int]]:
        """
        Executes the blocking UI render loop, allowing the user
        to draw the calibration line.

        Args:
            frame: The base image on which the calibration UI will be rendered.

        Returns:
            A list containing exactly two (x, y) tuples representing the finalized line.
        """

        self.frame = frame.copy()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        cv2.imshow(self.window_name, self.frame)
        cv2.setMouseCallback(self.window_name, self.handle_click)

        # UI Render Loop
        while True:
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

        cv2.destroyAllWindows()
        return self.line_points
