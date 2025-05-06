import argparse
import json
import os
import sys
import warnings
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from bd_metric.bjontegaard_metric import BD_RATE
from src.utils.common import create_folder


def parse_args(argv):
    parser = argparse.ArgumentParser(description='RD comparison for video codec models')

    parser.add_argument('--compare_between', type=str, default='class',
                        choices=['class', 'sequence'],
                        help='compare the performance between different classes/sequences')
    parser.add_argument('--base_method', type=str, required=True,
                        help='name of the anchor model')
    parser.add_argument('--log_paths', type=str, required=True, nargs='+',
                        help='list of model test result paths, model name followed by file path')
    parser.add_argument('--output_path', type=str, default='stdout',
                        help='print the results to console or save to file; TXT or CSV')
    parser.add_argument('--plot_path', type=str, default='',
                        help='path to save the plots')
    parser.add_argument('--plot_scheme', type=str, default=None,
                        choices=[None, 'combined', 'separate'], help='RD curve plot scheme')
    parser.add_argument('--distortion_metrics', type=str, nargs='+', default=['psnr'],
                        choices=['psnr', 'msssim', 'msssim_log'],
                        help='distortion metrics used to calculate BD-Rate and plot')
    parser.add_argument('--output_plot_data', type=str, default=None,
                        help='if given, save the plot data to csv file')
    parser.add_argument('--plot_rd_curve', type=int, default=1, choices=[0, 1],
                        help='if 1, plot RD curves')
    parser.add_argument('--auto_test', type=int, default=0, choices=[0, 1],
                        help='configurations for auto testing on AML')

    args = parser.parse_args(argv)
    return args


def _matplotlib_plt(data_dict, out_path, ds_name=None, distortion_metric='psnr'):
    plt.figure()
    for key in data_dict.keys():
        plt.plot(data_dict[key]['bpp'], data_dict[key][distortion_metric], label=key)
        plt.scatter(data_dict[key]['bpp'], data_dict[key][distortion_metric])

    if ds_name is not None:
        plt.gca().set_title(ds_name)
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.savefig(out_path)
    plt.close('all')


def matplotlib_plt(all_dataset_names, data_dict, out_prefix,
                   distortion_metric, plot_scheme='combined'):
    subplot_dicts = []
    for ds_name in all_dataset_names:
        ds_data_dict = {}
        for key, value in data_dict.items():
            if ds_name in value:
                ds_data_dict[key] = value[ds_name]
        if plot_scheme == 'separate':
            _matplotlib_plt(ds_data_dict,
                            f"{out_prefix}_{distortion_metric}_{ds_name}.png",
                            ds_name=ds_name,
                            distortion_metric=distortion_metric)
        else:
            subplot_dicts.append({'data_dict': ds_data_dict, 'ds_name': ds_name})

    if plot_scheme == 'combined':
        fig, axs = plt.subplots(1, len(subplot_dicts), figsize=(5*len(subplot_dicts), 4))
        if len(subplot_dicts) == 1:
            axs = [axs]
        for ax, d in zip(axs, subplot_dicts):
            data_dict = d['data_dict']
            ds_name = d['ds_name']
            for k in data_dict.keys():
                ax.plot(data_dict[k]['bpp'], data_dict[k][distortion_metric], label=k)
                ax.scatter(data_dict[k]['bpp'], data_dict[k][distortion_metric])

            ax.grid(True)
            ax.legend(loc='lower right')
            ax.set_title(ds_name)

        fig.tight_layout()
        fig.savefig(out_prefix+'_'+distortion_metric+'.png')


def rotate_results(ds_names, results, print_overall=False):
    new_ds_names = set()
    new_results = {}
    for ds in ds_names:
        new_results[ds] = {}
        for m in results:
            if ds in results[m]:
                new_results[ds][m] = results[m][ds]
            new_ds_names.add(m)
    if print_overall:
        new_results['* Overall'] = {}
        new_results['* Average'] = {}
        for m in results:
            if '* Overall' in results[m]:
                new_results['* Overall'][m] = results[m]['* Overall']
                new_results['* Average'][m] = results[m]['* Average']
    return list(new_ds_names), new_results


def avg_results(all_dataset_names, results):
    all_bd_rates = {}
    for method in sorted(list(results.keys())):
        for ds_name in all_dataset_names:
            if method not in all_bd_rates:
                all_bd_rates[method] = []
            if ds_name in results[method]:
                all_bd_rates[method].append(results[method][ds_name])
    avg_bd_rates = {}
    for key, value in all_bd_rates.items():
        if len(value) > 0:
            avg_bd_rates[key] = np.mean(value)
    return avg_bd_rates


def print_results(all_dataset_names, results, rotate=False, print_overall=False):
    if len(results.keys()) == 0:
        return

    if rotate:
        all_dataset_names, results = rotate_results(all_dataset_names, results, print_overall)

    all_dataset_names = sorted([x for x in all_dataset_names if not x.startswith('*')])
    all_method_names = [x for x in results.keys() if not x.startswith('*')]
    if print_overall and rotate:
        all_method_names += ['* Overall', '* Average']
    elif print_overall:
        all_dataset_names += ['* Overall', '* Average']
    method_name_l = max([len(x) for x in results.keys()]) + 2

    line = ' ' * method_name_l
    for x in all_dataset_names:
        line += ' {name:{width}}'.format(name=x, width=len(x)+2)
    print(line)

    for method in all_method_names:
        line = '{method:{method_name_width}}'.format(method=method, method_name_width=method_name_l)
        for ds_name in all_dataset_names:
            if ds_name not in results[method]:
                line += ' ' * (1 + len(ds_name)+2)
            else:
                line += ' {value:{width}}'.format(value='{:.1f}'.format(results[method][ds_name]),
                                                  width=len(ds_name)+2)
        print(line)


def print_results_different_metric(all_dataset_names, all_sequence_names, seq_results, results):
    for ds in sorted(all_dataset_names):
        print('-'*4, ds, '-'*4)
        print_seq_results = {}
        avg_bd_rates = avg_results(all_sequence_names[ds], seq_results)
        for m in sorted(list(results.keys())):
            if ds in results[m]:
                print_seq_results[m] = {}
                for seq in all_sequence_names[ds]:
                    if seq in seq_results[m]:
                        print_seq_results[m][seq] = seq_results[m][seq]
                    print_seq_results[m]['* Overall'] = results[m][ds]
                    print_seq_results[m]['* Average'] = avg_bd_rates[m]

        print_results(all_sequence_names[ds],
                      print_seq_results,
                      rotate=True,
                      print_overall=True)
    print()


def print_out(out, stdout, all_dataset_names, results):
    if out == 'stdout' or stdout:
        print_results(all_dataset_names, results)
    elif out.split('.')[-1] in ['csv', 'CSV']:
        raise NotImplementedError
    else:
        raise ValueError(f'unknown value for out: {out}')


def ssim_to_db(ssim):
    return -10 * np.math.log10(1 - ssim)


def mean_over_model(models):
    i_frame_num = 0
    p_frame_num = 0
    i_frame_bpp = 0.0
    i_frame_psnr = 0.0
    i_frame_msssim = 0.0
    p_frame_bpp = 0.0
    p_frame_psnr = 0.0
    p_frame_msssim = 0.0
    p_frame_bpp_mv_y = 0.0
    p_frame_bpp_mv_z = 0.0
    p_frame_bpp_y = 0.0
    p_frame_bpp_z = 0.0
    all_frame_bpp = 0.0
    all_frame_psnr = 0.0
    all_frame_msssim = 0.0
    for m in models:
        if 'ave_i_frame_msssim' not in m:
            m['ave_i_frame_msssim'] = 0
        if 'ave_p_frame_msssim' not in m:
            m['ave_p_frame_msssim'] = 0
        if 'ave_all_frame_msssim' not in m:
            m['ave_all_frame_msssim'] = 0
        if 'ave_i_frame_msssim_log' not in m:
            m['ave_i_frame_msssim_log'] = 0
        if 'ave_p_frame_msssim_log' not in m:
            m['ave_p_frame_msssim_log'] = 0
        if 'ave_all_frame_msssim_log' not in m:
            m['ave_all_frame_msssim_log'] = 0

        i_frame_num += m['i_frame_num']
        p_frame_num += m['p_frame_num']

        i_frame_bpp += m['ave_i_frame_bpp'] * m['i_frame_num']
        i_frame_psnr += m['ave_i_frame_psnr'] * m['i_frame_num']
        i_frame_msssim += m['ave_i_frame_msssim'] * m['i_frame_num']

        p_frame_bpp += m['ave_p_frame_bpp'] * m['p_frame_num']
        p_frame_psnr += m['ave_p_frame_psnr'] * m['p_frame_num']
        p_frame_msssim += m['ave_p_frame_msssim'] * m['p_frame_num']
        if 'ave_p_frame_bpp_mv_y' in m:
            p_frame_bpp_mv_y += m['ave_p_frame_bpp_mv_y'] * m['p_frame_num']
        if 'ave_p_frame_bpp_mv_z' in m:
            p_frame_bpp_mv_z += m['ave_p_frame_bpp_mv_z'] * m['p_frame_num']
        if 'ave_p_frame_bpp_y' in m:
            p_frame_bpp_y += m['ave_p_frame_bpp_y'] * m['p_frame_num']
        if 'ave_p_frame_bpp_z' in m:
            p_frame_bpp_z += m['ave_p_frame_bpp_z'] * m['p_frame_num']

        all_frame_num = m['p_frame_num'] + m['i_frame_num']
        all_frame_bpp += m['ave_all_frame_bpp'] * all_frame_num
        all_frame_psnr += m['ave_all_frame_psnr'] * all_frame_num
        all_frame_msssim += m['ave_all_frame_msssim'] * all_frame_num

    out_res = {}
    out_res['i_frame_num'] = i_frame_num
    out_res['p_frame_num'] = p_frame_num

    all_frame_num = i_frame_num + p_frame_num
    i_frame_num = 1 if i_frame_num == 0 else i_frame_num
    p_frame_num = 1 if p_frame_num == 0 else p_frame_num

    out_res['ave_i_frame_bpp'] = i_frame_bpp / i_frame_num
    out_res['ave_i_frame_psnr'] = i_frame_psnr / i_frame_num
    out_res['ave_i_frame_msssim'] = i_frame_msssim / i_frame_num
    out_res['ave_i_frame_msssim_log'] = ssim_to_db(out_res['ave_i_frame_msssim'])
    out_res['ave_p_frame_bpp'] = p_frame_bpp / p_frame_num
    out_res['ave_p_frame_psnr'] = p_frame_psnr / p_frame_num
    out_res['ave_p_frame_msssim'] = p_frame_msssim / p_frame_num
    out_res['ave_p_frame_msssim_log'] = ssim_to_db(out_res['ave_p_frame_msssim'])
    out_res['ave_p_frame_bpp_mv_y'] = p_frame_bpp_mv_y / p_frame_num
    out_res['ave_p_frame_bpp_mv_z'] = p_frame_bpp_mv_z / p_frame_num
    out_res['ave_p_frame_bpp_y'] = p_frame_bpp_y / p_frame_num
    out_res['ave_p_frame_bpp_z'] = p_frame_bpp_z / p_frame_num
    out_res['ave_all_frame_bpp'] = all_frame_bpp / all_frame_num
    out_res['ave_all_frame_psnr'] = all_frame_psnr / all_frame_num
    out_res['ave_all_frame_msssim'] = all_frame_msssim / all_frame_num
    out_res['ave_all_frame_msssim_log'] = ssim_to_db(out_res['ave_all_frame_msssim'])
    return out_res


def mean_over_sequence(res):
    new_res = {}  # model -> dataset -> [models]
    for m in res:
        new_res[m] = {}
        for d in res[m]:
            models = {}
            for s in res[m][d]:
                for model in res[m][d][s]:
                    if model['ckpt'] in models:
                        models[model['ckpt']].append(model)
                    else:
                        models[model['ckpt']] = [model]
            new_res[m][d] = []
            for _, value in models.items():
                new_res[m][d].append(mean_over_model(value))
    return new_res


def flatten_test_results(res):
    new_res = {}
    for key_method in res:
        new_res[key_method] = {}
        for ds_name in res[key_method]:
            for seq in res[key_method][ds_name]:
                ds_seq_name = seq
                new_res[key_method][ds_seq_name] = res[key_method][ds_name][seq]
    return new_res


def retrieve_data(json_dict, frame_type, base_method_name, distortion_metric):
    max_num_bitrate = 0
    data_dict = {}
    results = {}
    for key_method in json_dict.keys():
        data_dict[key_method] = {}
        for ds_name in json_dict[key_method]:
            data_dict[key_method][ds_name] = {}
            data_dict[key_method][ds_name]['bpp'] = []
            data_dict[key_method][ds_name]['psnr'] = []
            data_dict[key_method][ds_name]['msssim'] = []
            data_dict[key_method][ds_name]['msssim_log'] = []
            for one_data in json_dict[key_method][ds_name]:
                data_dict[key_method][ds_name]['bpp'].append(
                    one_data[f'ave_{frame_type}_frame_bpp'])
                data_dict[key_method][ds_name]['psnr'].append(
                    one_data[f'ave_{frame_type}_frame_psnr'])
                data_dict[key_method][ds_name]['msssim'].append(
                    one_data[f'ave_{frame_type}_frame_msssim'])
                data_dict[key_method][ds_name]['msssim_log'].append(
                    one_data[f'ave_{frame_type}_frame_msssim_log'])
            max_num_bitrate = max(max_num_bitrate, len(data_dict[key_method][ds_name]['bpp']))

    results[distortion_metric] = {}
    for key_method in json_dict.keys():
        if key_method == base_method_name:
            continue
        results[distortion_metric][key_method] = {}
        for ds_name in json_dict[key_method]:
            if (ds_name in data_dict[base_method_name]
                    and len(data_dict[key_method][ds_name]['bpp']) >= 3
                    and data_dict[base_method_name][ds_name]['bpp'][0] > 0
                    and data_dict[key_method][ds_name][distortion_metric][0] is not None
                    and data_dict[key_method][ds_name][distortion_metric][0] > 0):
                results[distortion_metric][key_method][ds_name] = BD_RATE(
                    data_dict[base_method_name][ds_name]['bpp'],
                    data_dict[base_method_name][ds_name][distortion_metric],
                    data_dict[key_method][ds_name]['bpp'],
                    data_dict[key_method][ds_name][distortion_metric], 1)

    return data_dict, results, max_num_bitrate


def compare(log_paths, base_method_name, compare_between, out, plot_path, plot_scheme,
            distortion_metric, output_plot_data, plot_rd_curve, auto_test):
    if output_plot_data:
        try:
            import pandas as pd  # pylint: disable=C0415
        except ImportError:
            raise ImportError('saving plot data to CSV requires Pandas library')

    stdout = None
    if out != 'stdout' and out.split('.')[-1] in ['txt', 'TXT']:
        postfix = out.split('.')[-1]
        if postfix in ['txt', 'TXT']:
            stdout = sys.stdout
            sys.stdout = open(out, 'a' if auto_test else 'w')  # pylint: disable=R1732

    if auto_test:
        print('Testing the following methods:')
        for key_method in log_paths:
            print(key_method, end=' ')
        print()

    all_dfs = {}
    json_dict = {}  # model -> dataset -> seq -> [models]
    ds_seq_names = {}
    seq_consistency = True
    for key_method in log_paths.keys():
        json_dict[key_method] = {}
        with open(log_paths[key_method]) as json_file:
            json_data = json.load(json_file)
            for ds_name in json_data.keys():
                json_dict[key_method][ds_name] = {}
                if ds_name not in ds_seq_names:
                    ds_seq_names[ds_name] = set(json_data[ds_name].keys())
                else:
                    if ds_seq_names[ds_name] != set(json_data[ds_name].keys()):
                        seq_consistency = False
                for seq in json_data[ds_name]:
                    json_dict[key_method][ds_name][seq] = []
                    for model_name in sorted(list(json_data[ds_name][seq].keys())):
                        json_dict[key_method][ds_name][seq].append(
                            json_data[ds_name][seq][model_name])
                        json_dict[key_method][ds_name][seq][-1]['ckpt'] = model_name

    if not seq_consistency:
        warnings.warn('inconsistency found in the sequences tested in each dataset')
    cls_json_dict = mean_over_sequence(json_dict)
    all_dataset_names = sorted(list(ds_seq_names.keys()))
    if compare_between == 'sequence':
        seq_json_dict = flatten_test_results(json_dict)
        all_sequence_names = {}
        for key, value in ds_seq_names.items():
            all_sequence_names[key] = []
            for seq in value:
                all_sequence_names[key].append(seq)

    for frame_type in ['all']:  # pylint: disable=too-many-nested-blocks
        print('='*12 + f'{frame_type:>3s} frame' + '='*12)
        frame_data, results_list, max_num_bitrate = retrieve_data(
                                                        cls_json_dict,
                                                        frame_type,
                                                        base_method_name,
                                                        distortion_metric)
        if compare_between == 'sequence':
            seq_frame_data, seq_results_list, \
                seq_max_num_bitrate = retrieve_data(
                    seq_json_dict,
                    frame_type,
                    base_method_name,
                    distortion_metric)

            print_results_different_metric(
                all_dataset_names,
                all_sequence_names,
                seq_results_list[distortion_metric],
                results_list[distortion_metric])
            frame_data, results_list[distortion_metric], max_num_bitrate = \
                seq_frame_data, seq_results_list[distortion_metric], seq_max_num_bitrate
        else:
            print_out(out, stdout, all_dataset_names, results_list[distortion_metric])

        if plot_rd_curve:
            if compare_between == 'sequence':
                names = []
                for _, value in all_sequence_names.items():
                    names += value
            else:
                names = all_dataset_names
            if plot_scheme == 'combined' and len(names) > 7:
                warnings.warn('plotting in combined mode with more than 7 datasets/sequences'
                              ' is not supported')
            else:
                matplotlib_plt(names, frame_data, os.path.join(plot_path, f'{frame_type}_frame'),
                               distortion_metric, plot_scheme=plot_scheme)
        if output_plot_data:
            models = sorted(list(frame_data.keys()))
            for m in models:
                for d in frame_data[m]:
                    if d+'_all_frame' not in all_dfs:
                        all_dfs[d+'_all_frame'] = pd.DataFrame(columns=[d+'_all_frame'])
                        all_dfs[d+'_all_frame'][d+'_all_frame'] = [x + 1
                                                                   for x in range(max_num_bitrate)]
                    all_dfs[d+'_all_frame'][m+'_bpp'] = pd.Series(frame_data[m][d]['bpp'])
                    all_dfs[d+'_all_frame'][m+'_psnr'] = pd.Series(frame_data[m][d]['psnr'])
                    all_dfs[d+'_all_frame'][m+'_msssim'] = pd.Series(frame_data[m][d]['msssim'])

    for key, value in all_dfs.items():
        with open(output_plot_data, 'a') as f:
            value.to_csv(f)

    if auto_test:
        print()

    if stdout:
        sys.stdout.close()
        sys.stdout = stdout


def main(argv):
    matplotlib.use('Agg')
    plt.rcParams.update({
        "grid.color": "0.5",
        "grid.linewidth": 0.5,
        "savefig.dpi": 300
    })

    args = parse_args(argv)

    if args.auto_test == 1:
        args.plot_rd_curve = 0
        warnings.warn('Plotting RD curves is disabled during auto test')

    base_method_name = args.base_method
    assert len(args.log_paths) % 2 == 0, \
        'log paths shoud include both the method name and the corresponding log path'
    log_paths = {}
    for i in range(len(args.log_paths)//2):
        log_paths[args.log_paths[2*i]] = args.log_paths[2*i+1]
    assert base_method_name in log_paths, 'log paths must include the base method'

    assert args.output_plot_data is None or args.output_plot_data[-4:] in ['.csv', '.CSV'], \
        'only CSV format supported for plot data output'

    if args.plot_scheme is None:
        args.plot_scheme = 'combined' if args.compare_between == 'class' else 'separate'
    if len(args.plot_path) > 0:
        create_folder(args.plot_path)
    for distortion_metric in args.distortion_metrics:
        print(f"result for: {distortion_metric.upper()}")
        compare(log_paths, base_method_name, args.compare_between, args.output_path, args.plot_path,
                args.plot_scheme, distortion_metric,
                args.output_plot_data, args.plot_rd_curve, bool(args.auto_test))


if __name__ == '__main__':
    main(sys.argv[1:])
