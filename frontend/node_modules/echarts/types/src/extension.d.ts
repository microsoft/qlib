import { registerPreprocessor, registerProcessor, registerPostInit, registerPostUpdate, registerAction, registerCoordinateSystem, registerLayout, registerVisual, registerLoading, registerMap, registerUpdateLifecycle } from './core/echarts.js';
import ComponentView from './view/Component.js';
import ChartView from './view/Chart.js';
import ComponentModel from './model/Component.js';
import SeriesModel from './model/Series.js';
import { Constructor } from './util/clazz.js';
import { SubTypeDefaulter } from './util/component.js';
import { registerImpl } from './core/impl.js';
import { registerPainter } from 'zrender/lib/zrender.js';
declare const extensionRegisters: {
    registerPreprocessor: typeof registerPreprocessor;
    registerProcessor: typeof registerProcessor;
    registerPostInit: typeof registerPostInit;
    registerPostUpdate: typeof registerPostUpdate;
    registerUpdateLifecycle: typeof registerUpdateLifecycle;
    registerAction: typeof registerAction;
    registerCoordinateSystem: typeof registerCoordinateSystem;
    registerLayout: typeof registerLayout;
    registerVisual: typeof registerVisual;
    registerTransform: typeof import("./data/helper/transform").registerExternalTransform;
    registerLoading: typeof registerLoading;
    registerMap: typeof registerMap;
    registerImpl: typeof registerImpl;
    PRIORITY: {
        PROCESSOR: {
            FILTER: number;
            SERIES_FILTER: number;
            STATISTIC: number;
        };
        VISUAL: {
            LAYOUT: number;
            PROGRESSIVE_LAYOUT: number;
            GLOBAL: number;
            CHART: number;
            POST_CHART_LAYOUT: number;
            COMPONENT: number;
            BRUSH: number;
            CHART_ITEM: number;
            ARIA: number;
            DECAL: number;
        };
    };
    ComponentModel: typeof ComponentModel;
    ComponentView: typeof ComponentView;
    SeriesModel: typeof SeriesModel;
    ChartView: typeof ChartView;
    registerComponentModel(ComponentModelClass: Constructor): void;
    registerComponentView(ComponentViewClass: typeof ComponentView): void;
    registerSeriesModel(SeriesModelClass: Constructor): void;
    registerChartView(ChartViewClass: typeof ChartView): void;
    registerSubTypeDefaulter(componentType: string, defaulter: SubTypeDefaulter): void;
    registerPainter(painterType: string, PainterCtor: Parameters<typeof registerPainter>[1]): void;
};
export declare type EChartsExtensionInstallRegisters = typeof extensionRegisters;
export declare type EChartsExtensionInstaller = (ec: EChartsExtensionInstallRegisters) => void;
export interface EChartsExtension {
    install: EChartsExtensionInstaller;
}
export declare function use(ext: EChartsExtensionInstaller | EChartsExtension | (EChartsExtensionInstaller | EChartsExtension)[]): void;
export declare type EChartsExtensionInstallerSimple = (registers: any) => void;
declare type SimpleEChartsExtensionType = EChartsExtensionInstallerSimple | {
    install: EChartsExtensionInstallerSimple;
};
export declare function useSimple(ext: SimpleEChartsExtensionType | (SimpleEChartsExtensionType)[]): void;
export {};
