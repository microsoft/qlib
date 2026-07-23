import ComponentModel from '../../model/Component.js';
import { AxisModelExtendedInCreator } from '../axisModelCreator.js';
import { AxisModelCommonMixin } from '../axisModelCommonMixin.js';
import { AxisBaseOption } from '../axisCommonTypes.js';
import AngleAxis from './AngleAxis.js';
import RadiusAxis from './RadiusAxis.js';
import { AxisBaseModel } from '../AxisBaseModel.js';
export declare type AngleAxisOption = AxisBaseOption & {
    mainType?: 'angleAxis';
    /**
     * Index of host polar component
     */
    polarIndex?: number;
    /**
     * Id of host polar component
     */
    polarId?: string;
    startAngle?: number;
    endAngle?: number;
    clockwise?: boolean;
    axisLabel?: AxisBaseOption['axisLabel'];
};
export declare type RadiusAxisOption = AxisBaseOption & {
    mainType?: 'radiusAxis';
    /**
     * Index of host polar component
     */
    polarIndex?: number;
    /**
     * Id of host polar component
     */
    polarId?: string;
};
declare type PolarAxisOption = AngleAxisOption | RadiusAxisOption;
declare class PolarAxisModel<T extends PolarAxisOption = PolarAxisOption> extends ComponentModel<T> implements AxisBaseModel<T> {
    static type: string;
    axis: AngleAxis | RadiusAxis;
    getCoordSysModel(): ComponentModel;
}
interface PolarAxisModel<T extends PolarAxisOption = PolarAxisOption> extends AxisModelCommonMixin<T>, AxisModelExtendedInCreator {
}
export { PolarAxisModel };
export declare class AngleAxisModel extends PolarAxisModel<AngleAxisOption> {
    static type: string;
    type: string;
    axis: AngleAxis;
}
export declare class RadiusAxisModel extends PolarAxisModel<RadiusAxisOption> {
    static type: string;
    type: string;
    axis: RadiusAxis;
}
