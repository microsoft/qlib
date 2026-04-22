declare type ColorFunc = (grad: number, fastMode: boolean, output: number[]) => void;
declare type ColorState = 'inRange' | 'outOfRange';
declare class HeatmapLayer {
    canvas: HTMLCanvasElement;
    blurSize: number;
    pointSize: number;
    maxOpacity: number;
    minOpacity: number;
    private _brushCanvas;
    private _gradientPixels;
    constructor();
    /**
     * Renders Heatmap and returns the rendered canvas
     * @param data array of data, each has x, y, value
     * @param width canvas width
     * @param height canvas height
     */
    update(data: number[][], width: number, height: number, normalize: (value: number) => number, colorFunc: Record<ColorState, ColorFunc>, isInRange?: (grad?: number) => boolean): HTMLCanvasElement;
    /**
     * get canvas of a black circle brush used for canvas to draw later
     */
    _getBrush(): HTMLCanvasElement;
    /**
     * get gradient color map
     * @private
     */
    _getGradient(colorFunc: Record<ColorState, ColorFunc>, state: ColorState): Uint8ClampedArray;
}
export default HeatmapLayer;
