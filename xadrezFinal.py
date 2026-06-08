import pygame
import chess
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import time
import json
from datetime import datetime

LARGURA_JANELA = 512
ALTURA_TABULEIRO = 512
TAMANHO_CASA = LARGURA_JANELA // 8

MARGEM_TOPO = 60
MARGEM_BASE = 60
ALTURA_TOTAL_JOGO = ALTURA_TABULEIRO + MARGEM_TOPO + MARGEM_BASE

ESTADO_TELA_INICIO = 0
ESTADO_MENU_PRINCIPAL = 1
ESTADO_MENU_TEMPO_1V1 = 2
ESTADO_MENU_DIFICULDADE_IA = 3
ESTADO_PERFIS_MANAGER = 4
ESTADO_CRIAR_PERFIL = 5
ESTADO_VER_HISTORICO = 6
ESTADO_SELECAO_PERFIL_1V1 = 7
ESTADO_SELECAO_PERFIL_IA = 8
ESTADO_JOGANDO = 9
ESTADO_PAUSA = 10
ESTADO_FIM_DE_JOGO = 11
ESTADO_ESTATISTICAS_PERFIL = 12

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PASTA_IMAGENS = os.path.join(BASE_DIR, "imagensxadrez")
PASTA_SONS = os.path.join(BASE_DIR, "sons")
ARQUIVO_DADOS = os.path.join(BASE_DIR, "dados_jogadores.json")

COR_BG_MODERNA = (49, 46, 43)
COR_BOTAO_MODERNO = (39, 36, 33)
COR_TEXTO_BOLD = (255, 255, 255)
COR_TEXTO_SUB = (180, 180, 180)
COR_DESTAQUE_MODERNO = (125, 169, 68)

COR_CASA_CLARA = (238, 238, 210)
COR_CASA_ESCURA = (118, 150, 86)
COR_DESTAQUE = (186, 202, 68)
COR_MOVIMENTO = (100, 111, 64, 150)
COR_CAPTURA = (255, 215, 0)
COR_VITORIA = (255, 215, 0)
COR_DERROTA = (220, 20, 60)
COR_EMPATE = (200, 200, 200)

DEVICE = torch.device("cpu")

TAM_ICONE_CAPTURA = 18  # tamanho em px do ícone pequeno de cada peça capturada
VALOR_PECA = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
}
CONTAGEM_INICIAL = {
    chess.PAWN: 8,
    chess.KNIGHT: 2,
    chess.BISHOP: 2,
    chess.ROOK: 2,
    chess.QUEEN: 1,
}
ORDEM_CAPTURAS = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]
_icones_captura = {} 


def carregar_dados():
    if os.path.exists(ARQUIVO_DADOS):
        try:
            with open(ARQUIVO_DADOS, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            pass
    return {"perfis": [], "historico": []}


def salvar_dados(dados):
    with open(ARQUIVO_DADOS, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=4)


class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 4096),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


def tabuleiro_para_tensor(board):
    t = np.zeros((12, 8, 8), dtype=np.float32)
    for square, peca in board.piece_map().items():
        canal = (peca.piece_type - 1) + (0 if peca.color == chess.WHITE else 6)
        t[canal, square // 8, square % 8] = 1.0
    return torch.from_numpy(t).unsqueeze(0).to(DEVICE)


def ia_escolher_movimento(modelo, board):
    moves = list(board.legal_moves)
    if not moves:
        return None
    with torch.no_grad():
        logits = modelo(tabuleiro_para_tensor(board))[0]
    return max(moves, key=lambda m: logits[m.from_square * 64 + m.to_square].item())


def carregar_imagens():
    pecas = {}
    mapa = {
        "P": "wP",
        "R": "wR",
        "N": "wN",
        "B": "wB",
        "Q": "wQ",
        "K": "wK",
        "p": "bP",
        "r": "bR",
        "n": "bN",
        "b": "bB",
        "q": "bQ",
        "k": "bK",
    }
    for k, v in mapa.items():
        caminho = os.path.join(PASTA_IMAGENS, f"{v}.png")
        if os.path.exists(caminho):
            img = pygame.image.load(caminho).convert_alpha()
            pecas[k] = pygame.transform.scale(img, (TAMANHO_CASA, TAMANHO_CASA))
    return pecas


def carregar_sons():
    sons = {}
    lista = {
        "move": "move.mp3",
        "capture": "capture.mp3",
        "win": "win.mp3",
        "loss": "loss.mp3",
        "notify": "notify.mp3",
    }
    if os.path.exists(PASTA_SONS):
        for chave, arquivo in lista.items():
            caminho = os.path.join(PASTA_SONS, arquivo)
            if os.path.exists(caminho):
                try:
                    sons[chave] = pygame.mixer.Sound(caminho)
                except:
                    pass
    return sons


def tocar_som_movimento(board, move, sons):
    board.push(move)
    xeque = board.is_check()
    board.pop()
    if xeque and "notify" in sons:
        sons["notify"].play()
    elif board.is_capture(move):
        if "capture" in sons:
            sons["capture"].play()
    elif "move" in sons:
        sons["move"].play()

def desenhar_botao_moderno(
    tela, titulo, subtitulo, x, y, w, h, hover, f_bold, f_regular
):
    pygame.draw.rect(
        tela,
        COR_DESTAQUE_MODERNO if hover else COR_BOTAO_MODERNO,
        (x, y, w, h),
        border_radius=8,
    )
    tela.blit(f_bold.render(titulo, True, COR_TEXTO_BOLD), (x + 20, y + 16))
    tela.blit(f_regular.render(subtitulo, True, COR_TEXTO_SUB), (x + 20, y + 40))
    return pygame.Rect(x, y, w, h)


def desenhar_botao_simples_moderno(tela, texto, x, y, w, h, hover, fonte):
    pygame.draw.rect(
        tela,
        COR_DESTAQUE_MODERNO if hover else COR_BOTAO_MODERNO,
        (x, y, w, h),
        border_radius=6,
    )
    txt = fonte.render(texto, True, COR_TEXTO_BOLD)
    tela.blit(txt, txt.get_rect(center=(x + w / 2, y + h / 2)))
    return pygame.Rect(x, y, w, h)


def desenhar_seletor(tela, titulo, valor, x, y, w, h, f_titulo, f_valor):
    tela.blit(f_titulo.render(titulo, True, COR_TEXTO_SUB), (x, y - 25))
    pygame.draw.rect(tela, COR_BOTAO_MODERNO, (x, y, w, h), border_radius=6)
    btn_esq = pygame.Rect(x, y, 40, h)
    btn_dir = pygame.Rect(x + w - 40, y, 40, h)
    pygame.draw.rect(tela, COR_DESTAQUE_MODERNO, btn_esq, border_radius=6)
    pygame.draw.rect(tela, COR_DESTAQUE_MODERNO, btn_dir, border_radius=6)
    tela.blit(f_valor.render("<", True, COR_TEXTO_BOLD), (x + 12, y + 8))
    tela.blit(f_valor.render(">", True, COR_TEXTO_BOLD), (x + w - 28, y + 8))
    txt = f_valor.render(valor, True, COR_TEXTO_BOLD)
    tela.blit(txt, txt.get_rect(center=(x + w / 2, y + h / 2)))
    return btn_esq, btn_dir


def confirmar_acao(tela, mensagem, fonte_bold, fonte_menu):
    fundo_atual = tela.copy()
    overlay = pygame.Surface((LARGURA_JANELA, ALTURA_TOTAL_JOGO))
    overlay.set_alpha(210)
    overlay.fill((0, 0, 0))
    fundo_atual.blit(overlay, (0, 0))

    rodando_confirmacao = True
    while rodando_confirmacao:
        mouse_pos = pygame.mouse.get_pos()
        tela.blit(fundo_atual, (0, 0))

        caixa_rect = pygame.Rect(56, ALTURA_TOTAL_JOGO // 2 - 80, 400, 160)
        pygame.draw.rect(tela, COR_BG_MODERNA, caixa_rect, border_radius=8)
        pygame.draw.rect(
            tela, COR_DESTAQUE_MODERNO, caixa_rect, width=2, border_radius=8
        )

        txt = fonte_bold.render(mensagem, True, COR_TEXTO_BOLD)
        tela.blit(
            txt, txt.get_rect(center=(LARGURA_JANELA // 2, ALTURA_TOTAL_JOGO // 2 - 30))
        )

        b_sim = desenhar_botao_simples_moderno(
            tela,
            "Sim",
            106,
            ALTURA_TOTAL_JOGO // 2 + 20,
            120,
            45,
            pygame.Rect(106, ALTURA_TOTAL_JOGO // 2 + 20, 120, 45).collidepoint(
                mouse_pos
            ),
            fonte_menu,
        )
        b_nao = desenhar_botao_simples_moderno(
            tela,
            "Não",
            286,
            ALTURA_TOTAL_JOGO // 2 + 20,
            120,
            45,
            pygame.Rect(286, ALTURA_TOTAL_JOGO // 2 + 20, 120, 45).collidepoint(
                mouse_pos
            ),
            fonte_menu,
        )

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                if b_sim.collidepoint(event.pos):
                    return True
                if b_nao.collidepoint(event.pos):
                    return False

def selecionar_promocao(tela, cor_jogador, imgs):
    opcoes = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    simbolos = (
        ["Q", "R", "B", "N"] if cor_jogador == chess.WHITE else ["q", "r", "b", "n"]
    )
    largura_box, altura_box = 320, 100
    x, y = (LARGURA_JANELA - largura_box) // 2, (ALTURA_TOTAL_JOGO - altura_box) // 2

    overlay = pygame.Surface((LARGURA_JANELA, ALTURA_TOTAL_JOGO))
    overlay.set_alpha(150)
    overlay.fill((0, 0, 0))
    tela.blit(overlay, (0, 0))

    pygame.draw.rect(
        tela, COR_BG_MODERNA, (x, y, largura_box, altura_box), border_radius=8
    )
    pygame.draw.rect(
        tela,
        COR_DESTAQUE_MODERNO,
        (x, y, largura_box, altura_box),
        width=3,
        border_radius=8,
    )

    rects = []
    for i, s in enumerate(simbolos):
        img = imgs[s]
        r = img.get_rect(center=(x + (i * 80) + 40, y + 50))
        tela.blit(img, r)
        rects.append((r, opcoes[i]))

    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                for r, tipo in rects:
                    if r.collidepoint(event.pos):
                        return tipo

def _obter_icones_captura(imagens):
    """Cache de ícones pequenos das peças (gerado 1x a partir das imagens grandes)."""
    if not _icones_captura:
        for simbolo, img in imagens.items():
            _icones_captura[simbolo] = pygame.transform.smoothscale(
                img, (TAM_ICONE_CAPTURA, TAM_ICONE_CAPTURA)
            )
    return _icones_captura


def _capturas_por_tipo(board, cor):
    """Quantas peças da cor dada já sumiram do tabuleiro (= foram capturadas)."""
    caps = {}
    for tipo in ORDEM_CAPTURAS:
        restantes = len(board.pieces(tipo, cor))
        caps[tipo] = max(0, CONTAGEM_INICIAL[tipo] - restantes)
    return caps


def _material(board, cor):
    """Soma do valor das peças da cor que ainda estão no tabuleiro."""
    return sum(VALOR_PECA[t] * len(board.pieces(t, cor)) for t in VALOR_PECA)


def desenhar_capturas(tela, imagens, board, x, y_centro, cor_capturada, vantagem):
    """Desenha, a partir de (x, y_centro), os ícones das peças da cor 'cor_capturada'
    que foram capturadas, agrupadas por tipo, e o '+N' de vantagem material (se >0)."""
    icones = _obter_icones_captura(imagens)
    caps = _capturas_por_tipo(board, cor_capturada)
    y = y_centro - TAM_ICONE_CAPTURA // 2
    cur_x = x
    passo = TAM_ICONE_CAPTURA - 7  # leve sobreposição entre peças do mesmo tipo
    for tipo in ORDEM_CAPTURAS:
        n = caps[tipo]
        if n == 0:
            continue
        simbolo = chess.Piece(tipo, cor_capturada).symbol()
        icone = icones.get(simbolo)
        if icone is None:
            continue
        for _ in range(n):
            tela.blit(icone, (cur_x, y))
            cur_x += passo
        cur_x += 5  # espaço extra entre grupos de tipos diferentes
    if vantagem > 0:
        f_vant = pygame.font.SysFont("Arial", 14, bold=True)
        tela.blit(
            f_vant.render(f"+{vantagem}", True, COR_TEXTO_SUB),
            (cur_x + 2, y_centro - 8),
        )


def desenhar_tudo(
    tela,
    board,
    imagens,
    selecao,
    fonte,
    nome_brancas,
    nome_pretas,
    t_brancas=None,
    t_pretas=None,
    peca_animando=None,
):
    pygame.draw.rect(tela, COR_BG_MODERNA, (0, 0, LARGURA_JANELA, MARGEM_TOPO))
    pygame.draw.rect(
        tela,
        COR_BG_MODERNA,
        (0, MARGEM_TOPO + ALTURA_TABULEIRO, LARGURA_JANELA, MARGEM_BASE),
    )

    f_nome = pygame.font.SysFont("Arial", 16, bold=True)
    y_pretas = MARGEM_TOPO // 2 - 10
    y_brancas = MARGEM_TOPO + ALTURA_TABULEIRO + MARGEM_BASE // 2 - 10
    s_pretas = f_nome.render(nome_pretas, True, COR_TEXTO_BOLD)
    s_brancas = f_nome.render(nome_brancas, True, COR_TEXTO_BOLD)
    tela.blit(s_pretas, (15, y_pretas))
    tela.blit(s_brancas, (15, y_brancas))

    # Peças capturadas ao lado de cada nome (estilo chess.com).
    # As pretas (em cima) capturaram peças BRANCAS; as brancas (embaixo), peças PRETAS.
    # Vantagem material só aparece pra quem está na frente.
    vantagem_brancas = _material(board, chess.WHITE) - _material(board, chess.BLACK)
    desenhar_capturas(
        tela, imagens, board,
        15 + s_pretas.get_width() + 10, y_pretas + s_pretas.get_height() // 2,
        chess.WHITE, -vantagem_brancas,
    )
    desenhar_capturas(
        tela, imagens, board,
        15 + s_brancas.get_width() + 10, y_brancas + s_brancas.get_height() // 2,
        chess.BLACK, vantagem_brancas,
    )

    for r in range(8):
        for c in range(8):
            cor = COR_CASA_CLARA if (r + c) % 2 == 0 else COR_CASA_ESCURA
            y_pos = r * TAMANHO_CASA + MARGEM_TOPO
            pygame.draw.rect(
                tela, cor, (c * TAMANHO_CASA, y_pos, TAMANHO_CASA, TAMANHO_CASA)
            )

            if c == 0:
                tela.blit(fonte.render(str(8 - r), True, (50, 50, 50)), (3, y_pos + 3))
            if r == 7:
                tela.blit(
                    fonte.render(chr(97 + c), True, (50, 50, 50)),
                    (c * TAMANHO_CASA + TAMANHO_CASA - 12, y_pos + TAMANHO_CASA - 15),
                )

    if selecao is not None:
        c, r = chess.square_file(selecao), 7 - chess.square_rank(selecao)
        pygame.draw.rect(
            tela,
            COR_DESTAQUE,
            (
                c * TAMANHO_CASA,
                r * TAMANHO_CASA + MARGEM_TOPO,
                TAMANHO_CASA,
                TAMANHO_CASA,
            ),
        )
        for m in board.legal_moves:
            if m.from_square == selecao:
                c_d, r_d = chess.square_file(m.to_square), 7 - chess.square_rank(
                    m.to_square
                )
                y_d = r_d * TAMANHO_CASA + MARGEM_TOPO
                if board.is_capture(m):
                    pygame.draw.rect(
                        tela,
                        COR_CAPTURA,
                        (c_d * TAMANHO_CASA, y_d, TAMANHO_CASA, TAMANHO_CASA),
                        4,
                    )
                else:
                    s = pygame.Surface((TAMANHO_CASA, TAMANHO_CASA), pygame.SRCALPHA)
                    pygame.draw.circle(
                        s,
                        COR_MOVIMENTO,
                        (TAMANHO_CASA // 2, TAMANHO_CASA // 2),
                        TAMANHO_CASA // 6,
                    )
                    tela.blit(s, (c_d * TAMANHO_CASA, y_d))

    for sq in chess.SQUARES:
        if peca_animando and sq == peca_animando["sq_origem"]:
            continue
        peca = board.piece_at(sq)
        if peca:
            tela.blit(
                imagens[peca.symbol()],
                (
                    chess.square_file(sq) * TAMANHO_CASA,
                    (7 - chess.square_rank(sq)) * TAMANHO_CASA + MARGEM_TOPO,
                ),
            )

    if t_brancas is not None and t_pretas is not None:
        f_t = pygame.font.SysFont("Courier", 24, bold=True)
        mb, sb = divmod(int(t_brancas), 60)
        mp, sp = divmod(int(t_pretas), 60)
        tela.blit(
            f_t.render(
                f"{mp:02}:{sp:02}",
                True,
                COR_TEXTO_BOLD if t_pretas > 30 else (255, 100, 100),
            ),
            (LARGURA_JANELA - 90, MARGEM_TOPO // 2 - 12),
        )
        pygame.draw.rect(
            tela,
            (255, 255, 255),
            (
                LARGURA_JANELA - 98,
                MARGEM_TOPO + ALTURA_TABULEIRO + MARGEM_BASE // 2 - 16,
                85,
                32,
            ),
            border_radius=4,
        )
        tela.blit(
            f_t.render(
                f"{mb:02}:{sb:02}", True, (0, 0, 0) if t_brancas > 30 else (200, 0, 0)
            ),
            (
                LARGURA_JANELA - 90,
                MARGEM_TOPO + ALTURA_TABULEIRO + MARGEM_BASE // 2 - 12,
            ),
        )


def animar_movimento(tela, board, imagens, move, fonte, nb, np, tb=None, tp=None):
    clock = pygame.time.Clock()
    c_s, r_s = chess.square_file(move.from_square), 7 - chess.square_rank(
        move.from_square
    )
    c_e, r_e = chess.square_file(move.to_square), 7 - chess.square_rank(move.to_square)
    peca_obj = board.piece_at(move.from_square)
    if not peca_obj:
        return
    img = imagens[peca_obj.symbol()]
    dx, dy = ((c_e - c_s) * TAMANHO_CASA) / 10, ((r_e - r_s) * TAMANHO_CASA) / 10
    for i in range(11):
        desenhar_tudo(
            tela,
            board,
            imagens,
            None,
            fonte,
            nb,
            np,
            tb,
            tp,
            peca_animando={"sq_origem": move.from_square},
        )
        tela.blit(
            img,
            (c_s * TAMANHO_CASA + dx * i, r_s * TAMANHO_CASA + MARGEM_TOPO + dy * i),
        )
        pygame.display.flip()
        clock.tick(60)


def exibir_fim_de_jogo(tela, txt, cor):
    fonte = pygame.font.SysFont("Arial", 32, bold=True)
    overlay = pygame.Surface((LARGURA_JANELA, ALTURA_TOTAL_JOGO))
    overlay.set_alpha(160)
    overlay.fill((0, 0, 0))
    tela.blit(overlay, (0, 0))
    txt_s = fonte.render(txt, True, cor)
    tela.blit(txt_s, txt_s.get_rect(center=(LARGURA_JANELA / 2, ALTURA_TOTAL_JOGO / 2)))

def main():
    pygame.init()
    pygame.font.init()
    pygame.mixer.init()
    tela = pygame.display.set_mode((LARGURA_JANELA, ALTURA_TOTAL_JOGO))
    pygame.display.set_caption("Chess Engine Profissional")

    fonte_bold = pygame.font.SysFont("Segoe UI", 22, bold=True)
    fonte_regular = pygame.font.SysFont("Segoe UI", 14)
    fonte_coord = pygame.font.SysFont("Arial", 14, bold=True)
    fonte_menu = pygame.font.SysFont("Arial", 20, bold=True)

    imgs = carregar_imagens()
    sons = carregar_sons()
    dados_app = carregar_dados()

    estado = ESTADO_TELA_INICIO
    modo_ia = False
    dificuldade = 1
    modelo = None
    board = chess.Board()
    tempo_base = 300
    tempo_brancas = tempo_pretas = 300
    ultimo_tempo = 0
    selecionado = None

    perfil_brancas = "Convidado"
    perfil_pretas = "Convidado"
    idx_brancas = 0
    idx_pretas = 0
    idx_stats = 0
    input_texto = ""
    nomes_difs = ["Iniciante", "Amador", "Intermediário", "Avançado", "Expert"]
    alerta_seletor = ""

    fim_timer = 0
    msg_fim = ""
    cor_fim = COR_EMPATE
    partida_registrada = False

    rodando = True
    while rodando:
        mouse_pos = pygame.mouse.get_pos()
        tela.fill(COR_BG_MODERNA)

        if estado == ESTADO_TELA_INICIO:
            tela.blit(
                pygame.font.SysFont("Courier New", 40, bold=True).render(
                    "CHESS ENGINE", True, COR_DESTAQUE_MODERNO
                ),
                (110, ALTURA_TOTAL_JOGO / 2 - 30),
            )
            tela.blit(
                pygame.font.SysFont("Arial", 16).render(
                    "Aperte qualquer tecla para começar", True, COR_TEXTO_BOLD
                ),
                (135, ALTURA_TOTAL_JOGO / 2 + 20),
            )
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type in (pygame.KEYDOWN, pygame.MOUSEBUTTONDOWN):
                    estado = ESTADO_MENU_PRINCIPAL

        elif estado == ESTADO_MENU_PRINCIPAL:
            tela.blit(
                pygame.font.SysFont("Arial", 28, bold=True).render(
                    "Menu Principal", True, COR_TEXTO_BOLD
                ),
                (160, 45),
            )
            b1 = desenhar_botao_moderno(
                tela,
                "1v1 Local",
                "Dois jogadores no mesmo PC",
                56,
                110,
                400,
                75,
                pygame.Rect(56, 110, 400, 75).collidepoint(mouse_pos),
                fonte_bold,
                fonte_regular,
            )
            b2 = desenhar_botao_moderno(
                tela,
                "Jogador x IA",
                "Desafie a Inteligência Artificial",
                56,
                200,
                400,
                75,
                pygame.Rect(56, 200, 400, 75).collidepoint(mouse_pos),
                fonte_bold,
                fonte_regular,
            )
            b3 = desenhar_botao_moderno(
                tela,
                "Perfis & Histórico",
                "Estatísticas, Cadastros e Logs",
                56,
                290,
                400,
                75,
                pygame.Rect(56, 290, 400, 75).collidepoint(mouse_pos),
                fonte_bold,
                fonte_regular,
            )
            b4 = desenhar_botao_moderno(
                tela,
                "Sair do Jogo",
                "Encerrar a aplicação",
                56,
                380,
                400,
                75,
                pygame.Rect(56, 380, 400, 75).collidepoint(mouse_pos),
                fonte_bold,
                fonte_regular,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if b1.collidepoint(event.pos):
                        estado = ESTADO_MENU_TEMPO_1V1
                        modo_ia = False
                    if b2.collidepoint(event.pos):
                        estado = ESTADO_MENU_DIFICULDADE_IA
                        modo_ia = True
                    if b3.collidepoint(event.pos):
                        estado = ESTADO_PERFIS_MANAGER
                    if b4.collidepoint(event.pos):
                        # NOVA JANELA DE CONFIRMAÇÃO AQUI
                        if confirmar_acao(
                            tela,
                            "Deseja realmente fechar o jogo?",
                            fonte_bold,
                            fonte_menu,
                        ):
                            rodando = False

        elif estado == ESTADO_PERFIS_MANAGER:
            tela.blit(
                fonte_bold.render("Gerenciar Jogadores", True, COR_TEXTO_BOLD),
                (150, 45),
            )
            b_criar = desenhar_botao_moderno(
                tela,
                "Criar Novo Perfil",
                "Adicionar ao catálogo local",
                56,
                110,
                400,
                75,
                pygame.Rect(56, 110, 400, 75).collidepoint(mouse_pos),
                fonte_bold,
                fonte_regular,
            )
            b_hist = desenhar_botao_moderno(
                tela,
                "Ver Logs & Histórico",
                "Ver os últimos 15 jogos salvos",
                56,
                200,
                400,
                75,
                pygame.Rect(56, 200, 400, 75).collidepoint(mouse_pos),
                fonte_bold,
                fonte_regular,
            )
            b_stats = desenhar_botao_moderno(
                tela,
                "Gerenciar Perfis",
                "Ver vitórias/derrotas e exclusão",
                56,
                290,
                400,
                75,
                pygame.Rect(56, 290, 400, 75).collidepoint(mouse_pos),
                fonte_bold,
                fonte_regular,
            )
            b_voltar = desenhar_botao_simples_moderno(
                tela,
                "Voltar",
                156,
                400,
                200,
                50,
                pygame.Rect(156, 400, 200, 50).collidepoint(mouse_pos),
                fonte_menu,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if b_criar.collidepoint(event.pos):
                        estado = ESTADO_CRIAR_PERFIL
                        input_texto = ""
                    if b_hist.collidepoint(event.pos):
                        estado = ESTADO_VER_HISTORICO
                        input_texto = ""
                    if b_stats.collidepoint(event.pos):
                        estado = ESTADO_ESTATISTICAS_PERFIL
                        idx_stats = 0
                    if b_voltar.collidepoint(event.pos):
                        estado = ESTADO_MENU_PRINCIPAL

        elif estado == ESTADO_ESTATISTICAS_PERFIL:
            tela.blit(
                fonte_bold.render("Estatísticas e Gerenciamento", True, COR_TEXTO_BOLD),
                (110, 50),
            )

            if len(dados_app["perfis"]) == 0:
                tela.blit(
                    fonte_regular.render(
                        "Nenhum perfil criado ainda.", True, COR_TEXTO_SUB
                    ),
                    (160, 200),
                )
                b_voltar = desenhar_botao_simples_moderno(
                    tela,
                    "Voltar",
                    156,
                    400,
                    200,
                    50,
                    pygame.Rect(156, 400, 200, 50).collidepoint(mouse_pos),
                    fonte_menu,
                )
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        rodando = False
                    if event.type == pygame.MOUSEBUTTONDOWN and b_voltar.collidepoint(
                        event.pos
                    ):
                        estado = ESTADO_PERFIS_MANAGER
            else:
                idx_stats = idx_stats % len(dados_app["perfis"])
                perfil_atual = dados_app["perfis"][idx_stats]
                be_s, bd_s = desenhar_seletor(
                    tela,
                    "Selecione o Perfil",
                    perfil_atual,
                    56,
                    130,
                    400,
                    50,
                    fonte_regular,
                    fonte_menu,
                )

                v_brancas = 0
                v_pretas = 0
                empates = 0
                derrotas = 0
                for h in dados_app["historico"]:
                    if h["brancas"] == perfil_atual or h["pretas"] == perfil_atual:
                        if h["resultado"] == "Empate":
                            empates += 1
                        elif f"({perfil_atual})" in h["resultado"]:
                            if h["brancas"] == perfil_atual:
                                v_brancas += 1
                            else:
                                v_pretas += 1
                        else:
                            derrotas += 1

                vitorias = v_brancas + v_pretas
                total = vitorias + derrotas + empates

                pygame.draw.rect(
                    tela, (30, 30, 30), (56, 210, 400, 110), border_radius=8
                )
                tela.blit(
                    fonte_bold.render(
                        f"Vitórias: {vitorias}", True, COR_DESTAQUE_MODERNO
                    ),
                    (80, 225),
                )
                tela.blit(
                    fonte_regular.render(
                        f"(Brancas: {v_brancas} | Pretas: {v_pretas})",
                        True,
                        COR_TEXTO_SUB,
                    ),
                    (200, 228),
                )
                tela.blit(
                    fonte_bold.render(f"Empates: {empates}", True, (200, 200, 200)),
                    (80, 255),
                )
                tela.blit(
                    fonte_bold.render(f"Derrotas: {derrotas}", True, COR_DERROTA),
                    (80, 285),
                )
                tela.blit(
                    fonte_regular.render(f"Total Jogado: {total}", True, COR_TEXTO_SUB),
                    (200, 285),
                )

                b_excluir = pygame.Rect(156, 350, 200, 45)
                cor_excluir = (
                    (200, 60, 60)
                    if b_excluir.collidepoint(mouse_pos)
                    else (150, 40, 40)
                )
                pygame.draw.rect(tela, cor_excluir, b_excluir, border_radius=6)
                txt_ex = fonte_menu.render("Excluir Perfil", True, COR_TEXTO_BOLD)
                tela.blit(txt_ex, txt_ex.get_rect(center=b_excluir.center))

                b_voltar = desenhar_botao_simples_moderno(
                    tela,
                    "Voltar",
                    156,
                    420,
                    200,
                    45,
                    pygame.Rect(156, 420, 200, 45).collidepoint(mouse_pos),
                    fonte_menu,
                )

                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        rodando = False
                    if event.type == pygame.MOUSEBUTTONDOWN:
                        if be_s.collidepoint(event.pos):
                            idx_stats = (idx_stats - 1) % len(dados_app["perfis"])
                        if bd_s.collidepoint(event.pos):
                            idx_stats = (idx_stats + 1) % len(dados_app["perfis"])
                        if b_voltar.collidepoint(event.pos):
                            estado = ESTADO_PERFIS_MANAGER
                        if b_excluir.collidepoint(event.pos):
                            # NOVA JANELA DE CONFIRMAÇÃO AQUI
                            if confirmar_acao(
                                tela,
                                f"Excluir o perfil '{perfil_atual}'?",
                                fonte_bold,
                                fonte_menu,
                            ):
                                dados_app["perfis"].remove(perfil_atual)
                                salvar_dados(dados_app)
                                idx_stats = 0

        elif estado == ESTADO_CRIAR_PERFIL:
            tela.blit(
                fonte_bold.render(
                    "Digite o Nome do Novo Perfil:", True, COR_TEXTO_BOLD
                ),
                (100, 150),
            )
            pygame.draw.rect(tela, (255, 255, 255), (56, 200, 400, 50), border_radius=5)
            tela.blit(
                fonte_bold.render(
                    input_texto + ("|" if time.time() % 1 > 0.5 else ""),
                    True,
                    (0, 0, 0),
                ),
                (65, 210),
            )
            b_salvar = desenhar_botao_simples_moderno(
                tela,
                "Salvar",
                90,
                320,
                150,
                45,
                pygame.Rect(90, 320, 150, 45).collidepoint(mouse_pos),
                fonte_menu,
            )
            b_cancelar = desenhar_botao_simples_moderno(
                tela,
                "Cancelar",
                270,
                320,
                150,
                45,
                pygame.Rect(270, 320, 150, 45).collidepoint(mouse_pos),
                fonte_menu,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if b_cancelar.collidepoint(event.pos):
                        estado = ESTADO_PERFIS_MANAGER
                    if b_salvar.collidepoint(event.pos) and input_texto.strip():
                        if (
                            input_texto.strip() not in dados_app["perfis"]
                            and input_texto.strip().lower() != "convidado"
                        ):
                            dados_app["perfis"].append(input_texto.strip())
                            salvar_dados(dados_app)
                        estado = ESTADO_PERFIS_MANAGER
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        estado = ESTADO_PERFIS_MANAGER
                    elif event.key == pygame.K_RETURN and input_texto.strip():
                        if (
                            input_texto.strip() not in dados_app["perfis"]
                            and input_texto.strip().lower() != "convidado"
                        ):
                            dados_app["perfis"].append(input_texto.strip())
                            salvar_dados(dados_app)
                        estado = ESTADO_PERFIS_MANAGER
                    elif event.key == pygame.K_BACKSPACE:
                        input_texto = input_texto[:-1]
                    elif len(input_texto) < 15:
                        input_texto += event.unicode

        elif estado == ESTADO_VER_HISTORICO:
            tela.blit(
                fonte_bold.render(
                    "Histórico Recente (Últimos 15)", True, COR_TEXTO_BOLD
                ),
                (110, 25),
            )
            pygame.draw.rect(tela, (255, 255, 255), (56, 75, 400, 30), border_radius=4)
            tela.blit(
                fonte_regular.render(
                    input_texto + ("|" if time.time() % 1 > 0.5 else ""),
                    True,
                    (0, 0, 0),
                ),
                (65, 80),
            )
            if not input_texto:
                tela.blit(
                    fonte_regular.render(
                        "Pesquise por jogador...", True, (120, 120, 120)
                    ),
                    (65, 80),
                )

            y_offset = 120
            filtrados = [
                h
                for h in reversed(dados_app["historico"])
                if input_texto.lower() in h["brancas"].lower()
                or input_texto.lower() in h["pretas"].lower()
            ]
            for h in filtrados[:15]:
                txt = f"{h['data']} | B: {h['brancas']} vs P: {h['pretas']} | {h['resultado']}"
                tela.blit(
                    pygame.font.SysFont("Arial", 12).render(txt, True, COR_TEXTO_BOLD),
                    (25, y_offset),
                )
                y_offset += 28

            b_voltar = desenhar_botao_simples_moderno(
                tela,
                "Voltar",
                156,
                560,
                200,
                45,
                pygame.Rect(156, 560, 200, 45).collidepoint(mouse_pos),
                fonte_menu,
            )
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN and b_voltar.collidepoint(
                    event.pos
                ):
                    estado = ESTADO_PERFIS_MANAGER
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_BACKSPACE:
                        input_texto = input_texto[:-1]
                    elif len(input_texto) < 20:
                        input_texto += event.unicode

        elif estado == ESTADO_MENU_TEMPO_1V1:
            tela.blit(
                fonte_bold.render("Selecione o Tempo", True, COR_TEXTO_BOLD), (160, 100)
            )
            bt1 = desenhar_botao_simples_moderno(
                tela,
                "5 Minutos",
                156,
                180,
                200,
                50,
                pygame.Rect(156, 180, 200, 50).collidepoint(mouse_pos),
                fonte_menu,
            )
            bt2 = desenhar_botao_simples_moderno(
                tela,
                "10 Minutos",
                156,
                260,
                200,
                50,
                pygame.Rect(156, 260, 200, 50).collidepoint(mouse_pos),
                fonte_menu,
            )
            b_voltar = desenhar_botao_simples_moderno(
                tela,
                "Voltar",
                156,
                360,
                200,
                50,
                pygame.Rect(156, 360, 200, 50).collidepoint(mouse_pos),
                fonte_menu,
            )
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if b_voltar.collidepoint(event.pos):
                        estado = ESTADO_MENU_PRINCIPAL
                    elif bt1.collidepoint(event.pos) or bt2.collidepoint(event.pos):
                        tempo_base = 300 if bt1.collidepoint(event.pos) else 600
                        tempo_brancas = tempo_pretas = tempo_base
                        estado = ESTADO_SELECAO_PERFIL_1V1
                        idx_brancas = 0
                        idx_pretas = 0
                        alerta_seletor = ""

        elif estado == ESTADO_MENU_DIFICULDADE_IA:
            tela.blit(fonte_bold.render("Nível da IA", True, COR_TEXTO_BOLD), (200, 40))
            b_dif = [
                desenhar_botao_simples_moderno(
                    tela,
                    f"Nível {i+1} ({nomes_difs[i]})",
                    126,
                    100 + (i * 55),
                    260,
                    40,
                    pygame.Rect(126, 100 + (i * 55), 260, 40).collidepoint(mouse_pos),
                    fonte_menu,
                )
                for i in range(5)
            ]
            b_voltar = desenhar_botao_simples_moderno(
                tela,
                "Voltar",
                126,
                400,
                260,
                40,
                pygame.Rect(126, 400, 260, 40).collidepoint(mouse_pos),
                fonte_menu,
            )
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if b_voltar.collidepoint(event.pos):
                        estado = ESTADO_MENU_PRINCIPAL
                    for i, b in enumerate(b_dif):
                        if b.collidepoint(event.pos):
                            dificuldade = i + 1
                            estado = ESTADO_SELECAO_PERFIL_IA
                            idx_brancas = 0
                            try:
                                modelo = ChessNet().to(DEVICE)
                                modelo.load_state_dict(
                                    torch.load(
                                        os.path.join(
                                            BASE_DIR, f"modelo_{dificuldade}.pth"
                                        ),
                                        map_location=DEVICE,
                                    )
                                )
                                modelo.eval()
                            except:
                                pass

        elif estado == ESTADO_SELECAO_PERFIL_1V1:
            perfis_disp = ["Convidado"] + dados_app["perfis"]
            tela.blit(
                fonte_bold.render("Selecione os Jogadores", True, COR_TEXTO_BOLD),
                (130, 60),
            )
            be_b, bd_b = desenhar_seletor(
                tela,
                "Peças Brancas",
                perfis_disp[idx_brancas],
                56,
                160,
                400,
                50,
                fonte_regular,
                fonte_menu,
            )
            be_p, bd_p = desenhar_seletor(
                tela,
                "Peças Pretas",
                perfis_disp[idx_pretas],
                56,
                260,
                400,
                50,
                fonte_regular,
                fonte_menu,
            )

            if alerta_seletor:
                tela.blit(
                    fonte_regular.render(alerta_seletor, True, COR_DERROTA), (100, 330)
                )

            b_start = desenhar_botao_simples_moderno(
                tela,
                "Iniciar Partida",
                56,
                380,
                400,
                50,
                pygame.Rect(56, 380, 400, 50).collidepoint(mouse_pos),
                fonte_menu,
            )
            b_voltar = desenhar_botao_simples_moderno(
                tela,
                "Voltar",
                56,
                450,
                400,
                50,
                pygame.Rect(56, 450, 400, 50).collidepoint(mouse_pos),
                fonte_menu,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if be_b.collidepoint(event.pos):
                        idx_brancas = (idx_brancas - 1) % len(perfis_disp)
                        alerta_seletor = ""
                    if bd_b.collidepoint(event.pos):
                        idx_brancas = (idx_brancas + 1) % len(perfis_disp)
                        alerta_seletor = ""
                    if be_p.collidepoint(event.pos):
                        idx_pretas = (idx_pretas - 1) % len(perfis_disp)
                        alerta_seletor = ""
                    if bd_p.collidepoint(event.pos):
                        idx_pretas = (idx_pretas + 1) % len(perfis_disp)
                        alerta_seletor = ""
                    if b_voltar.collidepoint(event.pos):
                        estado = ESTADO_MENU_TEMPO_1V1
                    if b_start.collidepoint(event.pos):
                        if (
                            perfis_disp[idx_brancas] == perfis_disp[idx_pretas]
                            and perfis_disp[idx_brancas] != "Convidado"
                        ):
                            alerta_seletor = (
                                "ERRO: Um perfil não pode jogar contra si mesmo!"
                            )
                        else:
                            if (
                                perfis_disp[idx_brancas] == "Convidado"
                                and perfis_disp[idx_pretas] == "Convidado"
                            ):
                                perfil_brancas, perfil_pretas = (
                                    "Convidado 1",
                                    "Convidado 2",
                                )
                            else:
                                perfil_brancas = perfis_disp[idx_brancas]
                                perfil_pretas = perfis_disp[idx_pretas]
                            estado = ESTADO_JOGANDO
                            board = chess.Board()
                            ultimo_tempo = pygame.time.get_ticks()
                            partida_registrada = False

        elif estado == ESTADO_SELECAO_PERFIL_IA:
            perfis_disp = ["Convidado"] + dados_app["perfis"]
            tela.blit(
                fonte_bold.render("Selecione seu Perfil", True, COR_TEXTO_BOLD),
                (150, 100),
            )
            be_b, bd_b = desenhar_seletor(
                tela,
                "Você (Brancas)",
                perfis_disp[idx_brancas],
                56,
                200,
                400,
                50,
                fonte_regular,
                fonte_menu,
            )
            b_start = desenhar_botao_simples_moderno(
                tela,
                "Iniciar Partida",
                56,
                330,
                400,
                50,
                pygame.Rect(56, 330, 400, 50).collidepoint(mouse_pos),
                fonte_menu,
            )
            b_voltar = desenhar_botao_simples_moderno(
                tela,
                "Voltar",
                56,
                400,
                400,
                50,
                pygame.Rect(56, 400, 400, 50).collidepoint(mouse_pos),
                fonte_menu,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if be_b.collidepoint(event.pos):
                        idx_brancas = (idx_brancas - 1) % len(perfis_disp)
                    if bd_b.collidepoint(event.pos):
                        idx_brancas = (idx_brancas + 1) % len(perfis_disp)
                    if b_voltar.collidepoint(event.pos):
                        estado = ESTADO_MENU_DIFICULDADE_IA
                    if b_start.collidepoint(event.pos):
                        perfil_brancas = perfis_disp[idx_brancas]
                        perfil_pretas = f"IA ({nomes_difs[dificuldade-1]})"
                        tempo_brancas = tempo_pretas = None
                        estado = ESTADO_JOGANDO
                        board = chess.Board()
                        ultimo_tempo = pygame.time.get_ticks()
                        partida_registrada = False

        elif estado == ESTADO_JOGANDO:
            if not modo_ia and tempo_brancas is not None and tempo_pretas is not None:
                agora = pygame.time.get_ticks()
                if agora - ultimo_tempo >= 1000:
                    if board.turn == chess.WHITE:
                        tempo_brancas -= 1
                    else:
                        tempo_pretas -= 1
                    ultimo_tempo = agora

                if tempo_brancas <= 0 or tempo_pretas <= 0:
                    estado = ESTADO_FIM_DE_JOGO
                    fim_timer = time.time()
                    msg_fim = (
                        "PRETAS VENCERAM POR TEMPO!"
                        if tempo_brancas <= 0
                        else "BRANCAS VENCERAM POR TEMPO!"
                    )
                    cor_fim = COR_DERROTA if tempo_brancas <= 0 else COR_VITORIA
                    if "loss" in sons:
                        sons["loss"].play()

            desenhar_tudo(
                tela,
                board,
                imgs,
                selecionado,
                fonte_coord,
                perfil_brancas,
                perfil_pretas,
                tempo_brancas,
                tempo_pretas,
            )

            if (
                modo_ia
                and board.turn == chess.BLACK
                and not board.is_game_over()
                and estado == ESTADO_JOGANDO
            ):
                pygame.display.set_caption("IA Pensando...")
                move_ia = ia_escolher_movimento(modelo, board)
                if move_ia:
                    animar_movimento(
                        tela,
                        board,
                        imgs,
                        move_ia,
                        fonte_coord,
                        perfil_brancas,
                        perfil_pretas,
                        tempo_brancas,
                        tempo_pretas,
                    )
                    tocar_som_movimento(board, move_ia, sons)
                    board.push(move_ia)
                pygame.display.set_caption("Chess Engine Profissional")

            if board.is_game_over():
                estado = ESTADO_FIM_DE_JOGO
                fim_timer = time.time()
                res = board.result()
                if res == "1-0":
                    msg_fim, cor_fim = "BRANCAS VENCERAM!", COR_VITORIA
                    sons.get("win", pygame.mixer.Sound(buffer=b"")).play()
                elif res == "0-1":
                    msg_fim, cor_fim = "PRETAS VENCERAM!", COR_DERROTA
                    sons.get("loss", pygame.mixer.Sound(buffer=b"")).play()
                else:
                    msg_fim, cor_fim = "EMPATE!", COR_EMPATE

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        estado = ESTADO_PAUSA
                if event.type == pygame.MOUSEBUTTONDOWN and board.turn == (
                    chess.WHITE if modo_ia else board.turn
                ):
                    x, y = event.pos
                    if MARGEM_TOPO <= y < MARGEM_TOPO + ALTURA_TABULEIRO:
                        sq = chess.square(
                            x // TAMANHO_CASA, 7 - ((y - MARGEM_TOPO) // TAMANHO_CASA)
                        )
                        if selecionado is None:
                            peca = board.piece_at(sq)
                            if peca and peca.color == board.turn:
                                selecionado = sq
                        else:
                            move = chess.Move(selecionado, sq)
                            move_teste = chess.Move(
                                selecionado, sq, promotion=chess.QUEEN
                            )

                            if (
                                move not in board.legal_moves
                                and move_teste in board.legal_moves
                            ):
                                escolha = selecionar_promocao(tela, board.turn, imgs)
                                move = chess.Move(selecionado, sq, promotion=escolha)
                                ultimo_tempo = pygame.time.get_ticks()

                            if move in board.legal_moves:
                                animar_movimento(
                                    tela,
                                    board,
                                    imgs,
                                    move,
                                    fonte_coord,
                                    perfil_brancas,
                                    perfil_pretas,
                                    tempo_brancas,
                                    tempo_pretas,
                                )
                                tocar_som_movimento(board, move, sons)
                                board.push(move)
                            selecionado = None

        elif estado == ESTADO_PAUSA:
            desenhar_tudo(
                tela,
                board,
                imgs,
                None,
                fonte_coord,
                perfil_brancas,
                perfil_pretas,
                tempo_brancas,
                tempo_pretas,
            )
            overlay = pygame.Surface((LARGURA_JANELA, ALTURA_TOTAL_JOGO))
            overlay.set_alpha(200)
            overlay.fill((0, 0, 0))
            tela.blit(overlay, (0, 0))

            tela.blit(
                fonte_bold.render("Jogo Pausado", True, COR_TEXTO_BOLD), (190, 140)
            )
            b_resume = desenhar_botao_simples_moderno(
                tela,
                "Retornar ao Jogo",
                126,
                220,
                260,
                45,
                pygame.Rect(126, 220, 260, 45).collidepoint(mouse_pos),
                fonte_menu,
            )
            b_restart = desenhar_botao_simples_moderno(
                tela,
                "Reiniciar Partida",
                126,
                280,
                260,
                45,
                pygame.Rect(126, 280, 260, 45).collidepoint(mouse_pos),
                fonte_menu,
            )
            b_quit = desenhar_botao_simples_moderno(
                tela,
                "Abandonar e Sair",
                126,
                340,
                260,
                45,
                pygame.Rect(126, 340, 260, 45).collidepoint(mouse_pos),
                fonte_menu,
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    rodando = False
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    estado = ESTADO_JOGANDO
                    ultimo_tempo = pygame.time.get_ticks()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if b_resume.collidepoint(event.pos):
                        estado = ESTADO_JOGANDO
                        ultimo_tempo = pygame.time.get_ticks()
                    if b_restart.collidepoint(event.pos):
                        # NOVA JANELA DE CONFIRMAÇÃO AQUI
                        if confirmar_acao(
                            tela, "Deseja reiniciar a partida?", fonte_bold, fonte_menu
                        ):
                            board = chess.Board()
                            tempo_brancas = tempo_pretas = (
                                tempo_base if not modo_ia else None
                            )
                            estado = ESTADO_JOGANDO
                            ultimo_tempo = pygame.time.get_ticks()
                            selecionado = None
                    if b_quit.collidepoint(event.pos):
                        # NOVA JANELA DE CONFIRMAÇÃO AQUI
                        if confirmar_acao(
                            tela, "Deseja abandonar a partida?", fonte_bold, fonte_menu
                        ):
                            estado = ESTADO_MENU_PRINCIPAL
                            board = chess.Board()
                            selecionado = None

        elif estado == ESTADO_FIM_DE_JOGO:
            desenhar_tudo(
                tela,
                board,
                imgs,
                None,
                fonte_coord,
                perfil_brancas,
                perfil_pretas,
                tempo_brancas,
                tempo_pretas,
            )
            exibir_fim_de_jogo(tela, msg_fim, cor_fim)

            if not partida_registrada:
                if "EMPATE" in msg_fim:
                    res_txt = "Empate"
                elif "BRANCAS" in msg_fim:
                    res_txt = f"Vitória B ({perfil_brancas})"
                else:
                    res_txt = f"Vitória P ({perfil_pretas})"

                dados_app["historico"].append(
                    {
                        "data": datetime.now().strftime("%d/%m %H:%M"),
                        "brancas": perfil_brancas,
                        "pretas": perfil_pretas,
                        "resultado": res_txt,
                    }
                )
                salvar_dados(dados_app)
                partida_registrada = True

            if time.time() - fim_timer > 3:
                b_menu = desenhar_botao_simples_moderno(
                    tela,
                    "Voltar ao Menu",
                    156,
                    ALTURA_TOTAL_JOGO - 100,
                    200,
                    50,
                    pygame.Rect(156, ALTURA_TOTAL_JOGO - 100, 200, 50).collidepoint(
                        mouse_pos
                    ),
                    fonte_menu,
                )
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        rodando = False
                    if event.type == pygame.MOUSEBUTTONDOWN and b_menu.collidepoint(
                        event.pos
                    ):
                        estado = ESTADO_MENU_PRINCIPAL
                        board = chess.Board()
            else:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        rodando = False

        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
