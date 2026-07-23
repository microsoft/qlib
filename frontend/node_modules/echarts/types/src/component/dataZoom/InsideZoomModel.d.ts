import DataZoomModel, { DataZoomOption } from './DataZoomModel.js';
export interface InsideDataZoomOption extends DataZoomOption {
    /**
     * Whether disable this inside zoom.
     */
    disabled?: boolean;
    /**
     * Whether disable zoom but only pan.
     */
    zoomLock?: boolean;
    zoomOnMouseWheel?: boolean | 'shift' | 'ctrl' | 'alt';
    moveOnMouseMove?: boolean | 'shift' | 'ctrl' | 'alt';
    moveOnMouseWheel?: boolean | 'shift' | 'ctrl' | 'alt';
    preventDefaultMouseMove?: boolean;
    /**
     * Inside dataZoom don't support textStyle
     */
    textStyle?: never;
}
declare class InsideZoomModel extends DataZoomModel<InsideDataZoomOption> {
    static readonly type = "dataZoom.inside";
    type: string;
    static defaultOption: InsideDataZoomOption;
}
export default InsideZoomModel;
