#-------------------------------------------------------------------------------
# - library
import	numpy				as		npy
import	pandas				as		pds
import	matplotlib
import	matplotlib.pyplot	as		plt
from	collections			import	deque
from	utils.ft_utils		import	(
		raw_filenames,
		filter_data,
		print_fname)
from	utils.ft_color		import	*

#-------------------------------------------------------------------------------
# eeg get function
def		get_eeg():
	print_fname(f"{MAG}[get_eeg]")

#-------------------------------------------------------------------------------
# replay function
def		replay(_eeg):
	RUN = [4, 8, 12]
	SBJ = [1]

	raw = filter_data


#-------------------------------------------------------------------------------
# main
if		__name__ == "__main__":
	plt.ioff()
	raw, eeg = get_eeg()
	plt.show()
	
	spectrum = raw.compute_psd()
	p = spectrum.plot_topomap()

	plt.ion()
	replay(eeg)