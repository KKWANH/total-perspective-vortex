# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: kkim <kkim@student.42.fr>                  +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2023/01/31 16:01:01 by kkim              #+#    #+#              #
#    Updated: 2023/06/21 06:21:26 by kkim             ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# ------------------------------------------------------------------------------
# Font Defines

# Color
BLA		= \033[30m
RED		= \033[31m
GRE		= \033[32m
YEL		= \033[33m
BLU		= \033[34m
MAG		= \033[35m
CYA		= \033[36m
WHI		= \033[37m
DEF		= \033[38m

# Font Style
BLD		= \033[1m
UND		= \033[4m

# Reset
RES		= \033[0m

# ------------------------------------------------------------------------------
# File Defines

SRC		= ./srcs
SETUP 	= ./setup/install.sh
ENV 	= ./ft_env/
ACTIV	= bin/activate

# ------------------------------------------------------------------------------
# Options
all:
	@make help

setup:
	@clear
	@make help
	@printf "\n"
	@$(SETUP) > /dev/null
	@printf "$(BLD)$(GRE)[Setup]     $(RES)\tSetting virtual-environment via $(CYA)[setup.sh]$(RES).\n"
	@printf "$(BLD)$(GRE)            $(RES)\t$(MAG)[pandas]$(RES)\n"
	@printf "$(BLD)$(GRE)            $(RES)\t$(MAG)[numpy]$(RES)\n"
	@printf "$(BLD)$(GRE)            $(RES)\t$(MAG)[matplotlib]$(RES)\n"
	@printf "$(BLD)$(GRE)            $(RES)\t$(MAG)[mne]$(RES)\n"
	@printf "$(BLD)$(GRE)            $(RES)\t$(MAG)[sklearn]$(RES)\n"
	@printf "$(BLD)$(GRE)[Setup]     $(RES)\tInstall has been compltetd.\n"
	@printf "$(BLD)$(GRE)            $(RES)\tPlease run $(CYA)[source ft_env/bin/activate]$(RES) to launch the virtual environment.\n"
	@printf "\n"

activate:
	@clear
	@make help
	@printf "\n"
	@printf "$(BLD)$(CYA)[Activate]  $(RES)\tPlease run $(CYA)[source ft_env/bin/activate]$(RES) to launch the virtual environment.\n"
	@printf "$(BLD)$(CYA)            $(RES)\tI'm sorry but running activate inside of Makefile doesn't work..\n"
	@printf "$(BLD)$(CYA)            $(RES)\tYou can deactivate it with $(MAG)[deactivate]$(RES) option/command.\n"
	@printf "\n"

deactivate:
	@clear
	@make help
	@printf "\n"
	@printf "$(BLD)$(MAG)[Activate]  $(RES)\tPlease run $(MAG)[deactivate]$(RES) to deactivate the virtual environment.\n"
	@printf "\n"

clean:
	@clear
	@make help
	@printf "\n"
	@printf "$(BLD)$(RED)[Clean]     $(RES)\tRemoving ft_env folder.\n"
	@rm -rf $(ENV)
	@printf "\n"

fit:
	@clear
	@make help
	@printf "\n"
	@printf "$(BLD)$(BLU)[fit]       $(RES)\tRunning src/ft_fit.py.\n"
	@python3 $(SRC)/ft_fit.py
	@printf "\n"

predict:
	@clear
	@make help
	@printf "\n"
	@printf "$(BLD)$(BLU)[predict]   $(RES)\tRunning src/ft_predict.py.\n"
	@python3 $(SRC)/ft_predict.py
	@printf "\n"

replay:
	@clear
	@make help
	@printf "\n"
	@printf "$(BLD)$(BLU)[replay]    $(RES)\tRunning src/ft_replay.py.\n"
	@python3 $(SRC)/ft_replay.py
	@printf "\n"

help:
	@clear
	@printf "\n"
	@printf "$(BLD)$(YEL)[Help]      $(RES)\tWelcome to $(BLD)$(UND)$(CYA)Total-Perpective-Vortex$(RES) by $(BLD)$(MAG)kkim, seongcho$(RES).\n"
	@printf "$(BLD)$(YEL)            $(RES)\tthere are $(YEL)6$(RES) options available.\n"
	@printf "\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(YEL)[help]$(RES)        makefile launch options\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(GRE)[setup]$(RES)       install libraries through python virtual environment\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(CYA)[activate]$(RES)    activating installed virtual environment\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(MAG)[deactivate]$(RES)  deactivating installed virtual environment\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(RED)[clean]$(RES)       remove ft_env(virtual environment) folder\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(BLU)[fit]$(RES)         fit model\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(BLU)[predict]$(RES)     predict with the model. you must have final_model.joblib file that was made by [fit]\n"
	@printf "$(BLD)$(YEL)            $(RES)\t$(BLD)$(BLU)[replay]$(RES)      replay the data\n"
	@printf "\n"

.PHONY: all help setup activate deactivate