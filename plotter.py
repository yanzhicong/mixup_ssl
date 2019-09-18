import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import csv
from yattag import Doc
from yattag import indent








class Plotter(object):
    def __init__(self):
        self._scalar_data_frame_dict = {}
        self._dist_data_frame_dict = {}


    def scalar(self, name, step, value, epoch=None):
        if epoch is not None:
            df = pd.DataFrame({'epoch' : epoch, 'step' : step, name : value, }, index=[0])
        else:
            df = pd.DataFrame({'step' : step, name : value, }, index=[0])

        if name not in self._scalar_data_frame_dict:
            self._scalar_data_frame_dict[name] = df
        else:
            self._scalar_data_frame_dict[name] = self._scalar_data_frame_dict[name].append(df, ignore_index=True)

    def dist(self, name, step, mean, var, epoch=None):
        if epoch is not None:
            df = pd.DataFrame({'epoch' : epoch, 'step' : step, name+'_mean' : mean, name+'_var' : var }, index=[0])
        else:
            df = pd.DataFrame({'step' : step, name+'_mean' : mean, name+'_var' : var, }, index=[0])

        if name not in self._dist_data_frame_dict:
            self._dist_data_frame_dict[name] = df
        else:
            self._dist_data_frame_dict[name] = self._dist_data_frame_dict[name].append(df, ignore_index=True)


    def dist2(self, name, step, value_list, epoch=None):
        mean = np.mean(value_list)
        var = np.var(value_list)
        self.dist(name, step, mean, var, epoch=epoch)


    def to_csv(self, output_dir):
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for name, data_frame in self._scalar_data_frame_dict.items():
            csv_filepath = os.path.join(output_dir, 'scalar_'+name+'.csv')
            print(csv_filepath)
            data_frame.to_csv(csv_filepath, index=False)
        for name, data_frame in self._dist_data_frame_dict.items():
            csv_filepath = os.path.join(output_dir, 'dist_'+name+'.csv')
            data_frame.to_csv(csv_filepath, index=False)

    def from_csv(self, output_dir):
        csv_name_list = [fn.split('.')[0] for fn in os.listdir(output_dir) if fn.endswith('csv')]
        for name in csv_name_list:
            if name.startswith('scalar_'):
                self._scalar_data_frame_dict[name[len('scalar_'):]] = pd.read_csv(os.path.join(output_dir, name+'.csv'))
            elif name.startswith('dists_'):
                self._dist_data_frame_dict[name[len('dists_'):]] = pd.read_csv(os.path.join(output_dir, name+'.csv'))


    def write_svg_all(self, output_dir):
        for ind, (name, data_frame) in enumerate(self._scalar_data_frame_dict.items()):
            output_svg_filepath = os.path.join(output_dir, name+'.svg')
            plt.figure()
            plt.clf()
            plt.plot(data_frame['step'], data_frame[name])
            plt.tight_layout()
            plt.savefig(output_svg_filepath)
            plt.close()

        for ind, (name, data_frame) in enumerate(self._dist_data_frame_dict.items()):
            output_svg_filepath = os.path.join(output_dir, name+'.svg')
            plt.figure()
            plt.clf()
            plt.errorbar(data_frame['step'], data_frame[name+'_mean'], yerr=data_frame[name+'_var'])
            plt.tight_layout()
            plt.savefig(output_svg_filepath)
            plt.close()


    def to_html_report(self, output_filepath):
        self.write_svg_all(os.path.dirname(output_filepath))
        doc, tag, text = Doc().tagtext()
        with open(output_filepath, 'w') as outfile:
            with tag('html'):
                with tag('body'):

                    with tag('h3'):
                        text('1. scalars')

                    for ind, (name, data_frame) in enumerate(self._scalar_data_frame_dict.items()):
                        with tag('h4', style='margin-left:20px'):
                            text('(%d). '%(ind+1)+name)
                        doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")

                    with tag('h3'):
                        text('2. distributions')

                    for ind, (name, data_frame) in enumerate(self._dist_data_frame_dict.items()):
                        with tag('h4', style='margin-left:20px'):
                            text('(%d). '%(ind+1)+name)
                        doc.stag("embed", style="width:800px;padding:5px;margin-left:20px", src=name+'.svg', type="image/svg+xml")

            result = indent(doc.getvalue())
            outfile.write(result)


class BatchPlotter(object):
    def __init__(self):
        pass


if __name__ == "__main__":
    p = Plotter()

    p.scalar('loss', 1, 100)
    p.scalar('loss', 2, 100)
    p.scalar('loss', 3, 100)
    p.scalar('loss', 4, 100)
    p.scalar('loss', 5, 100)
    p.scalar('loss', 6, 100)
    p.scalar('loss', 7, 100)
    p.scalar('loss', 8, 100)
    p.scalar('loss', 9, 100)
    p.scalar('loss', 10, 100)
    p.scalar('loss', 11, 100)
    p.scalar('loss', 12, 100)

    p.scalar('loss2', 1, 100)
    p.scalar('loss2', 2, 100)
    p.scalar('loss2', 3, 100)
    p.scalar('loss2', 4, 100)
    p.scalar('loss2', 5, 100)
    p.scalar('loss2', 6, 100)
    p.scalar('loss2', 7, 100)
    p.scalar('loss2', 8, 100)
    p.scalar('loss2', 9, 100)
    p.scalar('loss2', 10, 100)
    p.scalar('loss2', 11, 100)
    p.scalar('loss2', 12, 100)


    p.dist('loss3', 1, 100.0/1.0, 10)
    p.dist('loss3', 2, 100.0/2.0, 10)
    p.dist('loss3', 3, 100.0/3.0, 10)
    p.dist('loss3', 4, 100.0/4.0, 10)
    p.dist('loss3', 5, 100.0/5.0, 10)
    p.dist('loss3', 6, 100.0/6.0, 10)
    p.dist('loss3', 7, 100.0/7.0, 10)
    p.dist('loss3', 8, 100.0/8.0, 10)
    p.dist('loss3', 9, 100.0/9.0, 10)
    p.dist('loss3', 10, 100.0/10.0, 10)
    p.dist('loss3', 11, 100.0/11.0, 10)
    p.dist('loss3', 12, 100.0/12.0, 10)


    p.to_csv('./test_csv_output')
    p.to_html_report('./test_csv_output/output.html')
    # p.to_csv('./test_csv_output2')

