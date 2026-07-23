/// <reference types="react" />
import type { EChartsType } from 'echarts';
/**
 * Solve the type conflict caused by multiple type files
 */
export type EChartsOption = any;
export type EChartsInstance = EChartsType;
export type Opts = {
    readonly devicePixelRatio?: number;
    readonly renderer?: 'canvas' | 'svg';
    readonly width?: number | null | undefined | 'auto';
    readonly height?: number | null | undefined | 'auto';
    readonly locale?: string;
};
export type EChartsReactProps = React.HTMLAttributes<HTMLDivElement> & {
    /**
     * echarts library entry, use it for import necessary.
     */
    readonly echarts?: any;
    /**
     * echarts option
     */
    readonly option: EChartsOption;
    /**
     * echarts theme config, can be:
     * 1. theme name string
     * 2. theme object
     */
    readonly theme?: string | Record<string, any>;
    /**
     * notMerge config for echarts, default is `false`
     */
    readonly notMerge?: boolean;
    /**
     * replaceMerge config for echarts, default is `null`
     */
    readonly replaceMerge?: string | string[];
    /**
     * lazyUpdate config for echarts, default is `false`
     */
    readonly lazyUpdate?: boolean;
    /**
     * showLoading config for echarts, default is `false`
     */
    readonly showLoading?: boolean;
    /**
     * loadingOption config for echarts, default is `null`
     */
    readonly loadingOption?: any;
    /**
     * echarts opts config, default is `{}`
     */
    readonly opts?: Opts;
    /**
     * when after chart render, do the callback with echarts instance
     */
    readonly onChartReady?: (instance: EChartsInstance) => void;
    /**
     * bind events, default is `{}`
     */
    readonly onEvents?: Record<string, Function>;
    /**
     * should update echarts options
     */
    readonly shouldSetOption?: (prevProps: EChartsReactProps, props: EChartsReactProps) => boolean;
    /**
     * should trigger resize when window resize
     */
    readonly autoResize?: boolean;
};
