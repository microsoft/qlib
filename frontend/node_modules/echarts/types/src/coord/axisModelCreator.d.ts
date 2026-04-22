import ComponentModel from '../model/Component.js';
import OrdinalMeta from '../data/OrdinalMeta.js';
import { DimensionName, OrdinalRawValue } from '../util/types.js';
import { AxisBaseOption, CategoryAxisBaseOption } from './axisCommonTypes.js';
import { EChartsExtensionInstallRegisters } from '../extension.js';
declare type Constructor<T> = new (...args: any[]) => T;
export interface AxisModelExtendedInCreator {
    getCategories(rawData?: boolean): OrdinalRawValue[] | CategoryAxisBaseOption['data'];
    getOrdinalMeta(): OrdinalMeta;
}
/**
 * Generate sub axis model class
 * @param axisName 'x' 'y' 'radius' 'angle' 'parallel' ...
 */
export default function axisModelCreator<AxisOptionT extends AxisBaseOption, AxisModelCtor extends Constructor<ComponentModel<AxisOptionT>>>(registers: EChartsExtensionInstallRegisters, axisName: DimensionName, BaseAxisModelClass: AxisModelCtor, extraDefaultOption?: AxisOptionT): void;
export {};
